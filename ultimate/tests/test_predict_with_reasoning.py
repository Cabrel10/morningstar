import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Importations depuis le module à tester
from predict_with_reasoning import (
    generate_trading_insights,
    explain_signal,
    preprocess_data, # Sera mocké pour explain_signal
    create_model # Sera mocké pour explain_signal
)
from model.reasoning.reasoning_module import ExplanationDecoder # Sera mocké
from config.config import Config # Pour cfg factice

# --- Fixtures ---

@pytest.fixture
def mock_config():
    """Fixture pour un objet Config factice."""
    cfg = MagicMock(spec=Config)
    cfg.get_config.side_effect = lambda key, default=None: {
        "model.reasoning_architecture": {
            "num_reasoning_steps": 2, # Exemple
            "use_chain_of_thought": True, # Exemple
            "num_market_regime_classes": 2 # Exemple pour market_regime_expl_vec_i
        }
    }.get(key, default)
    return cfg

@pytest.fixture
def sample_feature_names():
    """Noms de features factices."""
    return ["feature1", "feature2", "feature3"]

@pytest.fixture
def sample_df_original_norm():
    """DataFrame original normalisé factice."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01"]),
        "symbol": ["BTC/USDT"],
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [102.0],
        "volume": [1000.0]
    })

@pytest.fixture
def sample_main_predictions_df():
    """DataFrame de prédictions principales factice."""
    # Doit inclure 'symbol' et 'close' car generate_trading_insights les utilise pour notify_trade_sync
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-01-01"]), # Ajouté pour la cohérence
        "symbol": ["BTC/USDT"], # Ajouté
        "close": [102.0], # Ajouté
        "market_regime_pred": [1], # Exemple: Haussier
        "sl_pred": [100.0],
        "tp_pred": [105.0],
        "risk_reward_ratio": [2.5],
        "market_regime_confidence": [0.8]
    })

@pytest.fixture
def sample_all_model_outputs():
    """Dictionnaire de toutes les sorties du modèle factice."""
    # Dimensions basées sur une seule instance (batch_size=1)
    # Les formes exactes (ex: (1, N) ou (N,)) dépendent de la sortie du modèle Keras.
    # Ici, on suppose (1, dim) pour la plupart, et une liste pour reasoning_steps.
    return {
        "market_regime": np.array([[0.1, 0.9]]), # (1, num_classes)
        "sl_tp": np.array([[100.0, 105.0]]),      # (1, 2)
        "signal": np.array([[0.1, 0.1, 0.6, 0.1, 0.1]]), # (1, num_signal_classes)
        "volatility_quantiles": np.array([[0.2, 0.7, 0.1]]),# (1, num_vol_classes)
        "final_reasoning": np.array([[0.1, 0.2, 0.3, 0.4]]), # (1, reasoning_units)
        "market_regime_explanation": np.array([[[0.1, 0.9], [0.2, 0.8]]]), # (1, num_regimes, reasoning_units)
        "sl_explanation": np.array([[0.5, 0.5, 0.1, 0.1]]),    # (1, reasoning_units)
        "tp_explanation": np.array([[0.3, 0.7, 0.2, 0.2]]),    # (1, reasoning_units)
        # Les étapes de raisonnement sont des sorties nommées individuellement par le modèle
        "reasoning_step_0": np.array([[0.1, 0.2, 0.1, 0.1]]), 
        "reasoning_step_1": np.array([[0.3, 0.4, 0.1, 0.1]]),
        # Ajouter plus d'étapes si num_reasoning_steps dans mock_config est > 2
        "attention_scores": np.array([[[0.2, 0.8, 0.5, 0.5]]]) # (1, num_heads, seq_len_q, seq_len_k) ou (1, dim_attention)
    }

# --- Tests ---

def test_generate_trading_insights(
    monkeypatch,
    mock_config,
    sample_main_predictions_df,
    sample_all_model_outputs,
    sample_df_original_norm,
    sample_feature_names
):
    """
    Teste generate_trading_insights pour vérifier l'appel à ExplanationDecoder
    et la mise à jour de la colonne 'reasoning'.
    """
    # 1. Mocker ExplanationDecoder
    mock_decoder_instance = MagicMock(spec=ExplanationDecoder)
    mock_decoder_instance.generate_chain_of_thought_explanation.return_value = "MOCK_COT_SUCCESS"
    
    # Patch l'instanciation de ExplanationDecoder pour retourner notre mock
    monkeypatch.setattr(
        "predict_with_reasoning.ExplanationDecoder", 
        lambda feature_names: mock_decoder_instance
    )

    # 2. Appeler la fonction à tester
    # La fonction modifie main_predictions_df en place et le retourne.
    result_df = generate_trading_insights(
        main_predictions_df=sample_main_predictions_df.copy(), # Passer une copie pour éviter modif de fixture
        all_model_outputs=sample_all_model_outputs,
        df_original_norm=sample_df_original_norm,
        feature_names=sample_feature_names,
        cfg=mock_config
    )

    # 3. Assertions
    # Vérifier que generate_chain_of_thought_explanation a été appelé une fois (car une seule ligne dans les données de test)
    assert mock_decoder_instance.generate_chain_of_thought_explanation.call_count == len(sample_main_predictions_df)
    
    # Récupérer les arguments du premier appel (et unique ici)
    call_args = mock_decoder_instance.generate_chain_of_thought_explanation.call_args
    
    # Vérifier certains des arguments passés (les plus importants)
    # Note: np.testing.assert_array_equal est bien pour comparer les arrays numpy
    assert call_args is not None, "generate_chain_of_thought_explanation n'a pas été appelé."
    
    # Vérifier market_data (premier argument positionnel ou nommé 'market_data')
    expected_market_data = sample_df_original_norm.iloc[0][['open', 'high', 'low', 'close', 'volume']].to_dict()
    assert call_args[1]['market_data'] == expected_market_data # Accès par nom de kwarg

    # Vérifier que les vecteurs de raisonnement sont passés
    # (on vérifie juste la présence des clés pour simplifier, les valeurs exactes sont dans sample_all_model_outputs)
    passed_kwargs = call_args[1] # Les kwargs sont dans le deuxième élément du tuple call_args (index 1)
    
    np.testing.assert_array_equal(passed_kwargs['final_reasoning_vec'], sample_all_model_outputs['final_reasoning'][0:1])
    
    # Pour market_regime_expl_vec, il faut simuler la sélection du régime prédit
    pred_idx = sample_main_predictions_df["market_regime_pred"].iloc[0]
    expected_market_regime_expl_vec = sample_all_model_outputs["market_regime_explanation"][0, pred_idx, :]
    np.testing.assert_array_equal(passed_kwargs['market_regime_expl_vec'], expected_market_regime_expl_vec)
    
    np.testing.assert_array_equal(passed_kwargs['sl_expl_vec'], sample_all_model_outputs['sl_explanation'][0:1])
    np.testing.assert_array_equal(passed_kwargs['tp_expl_vec'], sample_all_model_outputs['tp_explanation'][0:1])
    
    # Vérifier reasoning_steps_vecs (c'est une liste d'arrays)
    # Récupérer correctement la configuration imbriquée du mock
    reasoning_arch_cfg_mock = mock_config.get_config("model.reasoning_architecture")
    num_steps_config = reasoning_arch_cfg_mock.get("num_reasoning_steps")
    use_cot_flag_config = reasoning_arch_cfg_mock.get("use_chain_of_thought")

    if passed_kwargs['reasoning_steps_vecs'] is not None:
        assert len(passed_kwargs['reasoning_steps_vecs']) == num_steps_config
        for j in range(num_steps_config):
            step_key = f"reasoning_step_{j}"
            np.testing.assert_array_equal(passed_kwargs['reasoning_steps_vecs'][j], sample_all_model_outputs[step_key][0:1])
    else:
        # Si c'est None, cela signifie que use_cot_flag était False ou num_reasoning_steps était 0
        assert not use_cot_flag_config or num_steps_config == 0
        
    np.testing.assert_array_equal(passed_kwargs['attention_scores_vec'], sample_all_model_outputs['attention_scores'][0:1])

    # Vérifier que la colonne 'reasoning' a été mise à jour
    assert "reasoning" in result_df.columns
    assert all(result_df["reasoning"] == "MOCK_COT_SUCCESS")


def test_explain_signal(
    monkeypatch,
    mock_config,
    sample_df_original_norm, # Utilisé comme one_row_df
    sample_feature_names,
    sample_all_model_outputs # Utilisé pour la sortie mockée de model.predict
):
    """
    Teste la fonction explain_signal.
    Vérifie que preprocess_data, create_model.predict, et ExplanationDecoder sont appelés correctement.
    """
    # 1. Mocker preprocess_data
    mock_X_signal = {"technical_input": np.array([[0.1, 0.2]])}
    mock_df_norm_signal = sample_df_original_norm.copy()
    mock_actual_feature_names = sample_feature_names
    
    # Créer un mock et le garder pour les assertions
    mock_preprocess_func = MagicMock(return_value=(mock_X_signal, mock_df_norm_signal, mock_actual_feature_names))
    monkeypatch.setattr("predict_with_reasoning.preprocess_data", mock_preprocess_func)

    # 2. Mocker create_model et sa méthode predict
    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = sample_all_model_outputs
    
    # Créer un mock pour create_model et le garder
    mock_create_model_func = MagicMock(return_value=mock_model_instance)
    monkeypatch.setattr("predict_with_reasoning.create_model", mock_create_model_func)

    # 3. Mocker ExplanationDecoder
    mock_decoder_instance = MagicMock(spec=ExplanationDecoder)
    mock_decoder_instance.generate_chain_of_thought_explanation.return_value = "MOCK_EXPLAIN_SUCCESS"
    monkeypatch.setattr(
        "predict_with_reasoning.ExplanationDecoder",
        lambda feature_names: mock_decoder_instance # Le lambda retourne le mock_decoder_instance qu'on peut vérifier
    )

    # 4. Appeler la fonction à tester
    result_explanation = explain_signal(
        signal_data_df=sample_df_original_norm.copy(),
        cfg=mock_config,
        feature_names_list=sample_feature_names
    )

    # 5. Assertions
    mock_preprocess_func.assert_called_once()
    # On pourrait vérifier les args de mock_preprocess_func si nécessaire

    mock_create_model_func.assert_called_once_with(mock_X_signal, mock_actual_feature_names, mock_config)
    mock_model_instance.predict.assert_called_once_with(mock_X_signal)
    
    # Vérifier que ExplanationDecoder a été instancié avec les bons feature_names
    # (le patch de setattr gère cela, on vérifie l'appel à sa méthode)

    # Vérifier que generate_chain_of_thought_explanation a été appelé
    mock_decoder_instance.generate_chain_of_thought_explanation.assert_called_once()
    call_args_explain = mock_decoder_instance.generate_chain_of_thought_explanation.call_args
    assert call_args_explain is not None, "generate_chain_of_thought_explanation (dans explain_signal) n'a pas été appelé."

    # Vérifier les arguments passés à generate_chain_of_thought_explanation
    passed_kwargs_explain = call_args_explain[1]
    expected_market_data_explain = sample_df_original_norm.iloc[0][['open', 'high', 'low', 'close', 'volume']].to_dict()
    assert passed_kwargs_explain['market_data'] == expected_market_data_explain
    
    # Vérifier quelques vecteurs de raisonnement (similaire au test précédent)
    np.testing.assert_array_equal(passed_kwargs_explain['final_reasoning_vec'], sample_all_model_outputs['final_reasoning'][0:1])
    # ... (on pourrait ajouter plus d'assertions sur les kwargs si nécessaire)

    # Vérifier la valeur retournée par explain_signal
    assert result_explanation == "MOCK_EXPLAIN_SUCCESS"
