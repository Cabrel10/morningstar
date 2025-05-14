#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour le module model/training/enhanced_reasoning_training.py
"""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import pandas as pd
import tensorflow as tf

# Importer la fonction main à tester
from model.training.enhanced_reasoning_training import main

# --- Fixtures ---


@pytest.fixture
def mock_args(tmp_path):
    """Mocker les arguments de la ligne de commande."""
    args = MagicMock()
    args.data_path = str(tmp_path / "dummy_data.parquet")  # Chemin vers un fichier de données factice
    args.output_dir = str(tmp_path / "test_output_enhanced")
    # Créer un fichier de données factice pour que load_and_split_data ne lève pas FileNotFoundError
    pd.DataFrame({"feature_1": [1, 2, 3], "market_regime": [0, 1, 0]}).to_parquet(args.data_path)
    return args


@pytest.fixture
def mock_config_enhanced(mocker):
    """Mocker la classe Config pour les tests d'enhanced_reasoning_training."""
    mock = mocker.patch("model.training.enhanced_reasoning_training.Config", autospec=True)
    instance = mock.return_value

    # Valeurs de configuration de base pour les tests
    config_values = {
        "model.num_technical": 1,
        "model.num_mcp": 0,
        "model.num_cryptobert": 0,
        "model.num_hmm": 0,
        "model.num_sentiment": 0,
        "model.num_market": 0,
        "model.instrument_vocab_size": 0,
        "data.label_columns": ["market_regime"],
        "model.reasoning_architecture": {
            "instrument_embedding_dim": 8,  # Valeur par défaut si non utilisée
            "num_sl_tp_outputs": 2,
            "l2_reg": 0.001,
            "dropout_rate": 0.3,
            "use_batch_norm": True,
            "num_reasoning_steps": 3,
            "reasoning_units": 128,
            # Les dimensions d'entrée seront ajoutées dynamiquement dans le test/main
        },
        "model.active_outputs": ["market_regime"],
        "data.label_mappings.market_regime": {0: 0, 1: 1},  # Mapping simple
        "training.learning_rate": 0.001,
        "training.validation_split": 0.1,
        "training.epochs": 1,  # Entraînement très court pour les tests
        "training.batch_size": 32,
    }

    instance.get_config.side_effect = lambda key, default=None: config_values.get(key, default)
    return instance


@pytest.fixture
def mock_load_and_split_data(mocker):
    """Mocker la fonction load_and_split_data."""
    mock = mocker.patch("model.training.enhanced_reasoning_training.load_and_split_data")
    # Simuler le retour de X (dictionnaire de features) et y (dictionnaire de labels)
    # X doit contenir les clés attendues par build_reasoning_model si elles sont utilisées pour déduire les shapes
    # Pour ce test, on suppose que les dimensions sont fournies par la config
    X_mock = {
        "technical_input": np.random.rand(10, 1).astype(np.float32),  # 1 feature technique
        # Ajouter d'autres inputs si num_* > 0 dans mock_config_enhanced
    }
    y_mock = {"market_regime": pd.Series(np.random.randint(0, 2, size=10))}
    mock.return_value = (X_mock, y_mock)
    return mock


@pytest.fixture
def mock_build_reasoning_model(mocker):
    """Mocker la fonction build_reasoning_model."""
    mock_model_instance = MagicMock(spec=tf.keras.Model)
    mock_model_instance.output_names = ["market_regime"]  # Simuler une sortie
    mock_model_instance.fit.return_value = MagicMock()  # Simuler l'historique d'entraînement

    mock = mocker.patch(
        "model.training.enhanced_reasoning_training.build_reasoning_model", return_value=mock_model_instance
    )
    return mock


@pytest.fixture
def mock_compile_reasoning_model(mocker):
    """Mocker la fonction compile_reasoning_model."""
    return mocker.patch("model.training.enhanced_reasoning_training.compile_reasoning_model")


@pytest.fixture
def mock_explanation_decoder(mocker):
    """Mocker la classe ExplanationDecoder."""
    return mocker.patch("model.training.enhanced_reasoning_training.ExplanationDecoder", autospec=True)


# --- Tests pour la fonction main ---


def test_main_pipeline_execution(
    mock_args,
    mock_config_enhanced,
    mock_load_and_split_data,
    mock_build_reasoning_model,
    mock_compile_reasoning_model,
    mock_explanation_decoder,  # Ajouter le mock ici
):
    """Teste l'exécution de base du pipeline main()."""

    # Exécuter la fonction main
    with patch.object(
        sys, "argv", ["script_name", "--data-path", mock_args.data_path, "--output-dir", mock_args.output_dir]
    ):
        main()

    # Vérifier les appels aux fonctions mockées
    mock_load_and_split_data.assert_called_once_with(
        file_path=mock_args.data_path,
        label_columns=["market_regime"],
        as_tensor=False,
        num_technical_features=1,  # De mock_config_enhanced
        num_llm_features=0,  # De mock_config_enhanced
        num_mcp_features=0,  # De mock_config_enhanced
    )

    mock_build_reasoning_model.assert_called_once()
    # Vérifier certains arguments clés passés à build_reasoning_model
    # build_reasoning_model est appelé avec **model_params, donc les args sont dans kwargs
    called_kwargs = mock_build_reasoning_model.call_args.kwargs
    assert called_kwargs["tech_input_shape"] == (1,)  # (num_technical,)
    assert called_kwargs["num_market_regime_classes"] == 2  # Déduit du mapping
    assert "market_regime" in called_kwargs["active_outputs"]

    mock_compile_reasoning_model.assert_called_once()
    # compile_reasoning_model est appelé avec le modèle comme arg positionnel,
    # et le reste en kwargs.
    compile_pos_args = mock_compile_reasoning_model.call_args.args
    compile_kwargs = mock_compile_reasoning_model.call_args.kwargs
    assert compile_pos_args[0] == mock_build_reasoning_model.return_value  # Le modèle
    assert compile_kwargs["learning_rate"] == 0.001
    assert "market_regime" in compile_kwargs["active_outputs"]

    # Vérifier l'appel à model.fit()
    mock_model_instance = mock_build_reasoning_model.return_value
    mock_model_instance.fit.assert_called_once()
    fit_args, fit_kwargs = mock_model_instance.fit.call_args

    # Vérifier les données passées à fit (X et y)
    # X_passed_to_fit = fit_args[0]
    y_passed_to_fit = fit_args[1]

    # assert X_passed_to_fit == mock_load_and_split_data.return_value[0] # X_mock
    assert "market_regime" in y_passed_to_fit
    assert np.array_equal(
        y_passed_to_fit["market_regime"], mock_load_and_split_data.return_value[1]["market_regime"].values
    )

    assert fit_kwargs["validation_split"] == 0.1
    assert fit_kwargs["epochs"] == 1
    assert fit_kwargs["batch_size"] == 32
    assert len(fit_kwargs["callbacks"]) == 3  # ModelCheckpoint, EarlyStopping, TensorBoard

    # Vérifier l'instanciation de ExplanationDecoder
    mock_explanation_decoder.assert_called_once_with(feature_names=[])


def test_main_handles_missing_config_values(
    mock_args,
    mock_config_enhanced,  # Utilise la config de base
    mock_load_and_split_data,
    mock_build_reasoning_model,
    mock_compile_reasoning_model,
    mock_explanation_decoder,
):
    """Teste que main() utilise des valeurs par défaut si certaines configs sont absentes."""

    # Modifier le side_effect pour simuler des clés manquantes
    original_side_effect = mock_config_enhanced.get_config.side_effect

    def new_side_effect(key, default=None):
        if key == "training.epochs":  # Simuler epochs manquant
            return default
        if key == "data.label_mappings.market_regime":  # Simuler mapping manquant
            return default
        return original_side_effect(key, default)

    mock_config_enhanced.get_config.side_effect = new_side_effect

    with patch.object(
        sys, "argv", ["script_name", "--data-path", mock_args.data_path, "--output-dir", mock_args.output_dir]
    ):
        main()

    # Vérifier que build_reasoning_model a été appelé avec num_market_regime_classes=2 par défaut
    called_kwargs_missing = mock_build_reasoning_model.call_args.kwargs
    assert called_kwargs_missing["num_market_regime_classes"] == 2

    # Vérifier que model.fit a été appelé avec epochs=100 par défaut (valeur par défaut dans le code de main)
    # Note: la valeur par défaut dans le code de main() est 100, pas celle de la fixture.
    # Il faudrait mocker cfg.get_config('training.epochs', 100) pour que le test soit plus précis.
    # Pour l'instant, on vérifie juste que fit a été appelé.
    mock_build_reasoning_model.return_value.fit.assert_called_once()
    _, fit_kwargs = mock_build_reasoning_model.return_value.fit.call_args
    assert fit_kwargs["epochs"] == 100  # Valeur par défaut dans main()


# Ajouter d'autres tests, par exemple pour vérifier la création du répertoire de sortie,
# la gestion des erreurs de chargement de données, etc.
