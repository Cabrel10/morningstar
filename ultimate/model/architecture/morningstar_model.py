import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Tuple, Union
from .enhanced_hybrid_model import build_enhanced_hybrid_model


class MorningstarModel:
    """Wrapper pour intégrer le modèle hybride dans le workflow de trading."""

    def __init__(
        self, model_config: Dict = None, use_llm: bool = True, llm_fallback_strategy: str = "technical_projection"
    ):
        """
        Initialise le wrapper du modèle.

        Args:
            model_config: Configuration du modèle (peut être chargée depuis un fichier YAML)
            use_llm: Indique si les données LLM sont disponibles pour l'entraînement/inférence
            llm_fallback_strategy: Stratégie de fallback quand les données LLM ne sont pas disponibles
                - 'zero_vector': Utilise un vecteur de zéros à la place des embeddings LLM
                - 'learned_embedding': Utilise un embedding appris pendant l'entraînement
                - 'technical_projection': Projette les données techniques dans l'espace LLM
        """
        self.logger = logging.getLogger("MorningstarModel")
        self.use_llm = use_llm
        self.llm_fallback_strategy = llm_fallback_strategy

        self.config = model_config or {
            "num_technical_features": 38,
            "llm_embedding_dim": 768,
            "mcp_input_dim": 128,  # Assuming default MCP dim
            "hmm_input_dim": 4,  # Default HMM dim (regime + 3 probs)
            "instrument_vocab_size": 10,  # Assuming default vocab size
            "instrument_embedding_dim": 8,  # Assuming default embedding dim
            "num_signal_classes": 5,
            "num_volatility_regimes": 3,  # This might be deprecated if volatility_quantiles is used
            "num_market_regimes": 4,
            "num_volatility_quantiles": 3,  # Add this if used by enhanced model
            "num_sl_tp_outputs": 2,  # Add this if used by enhanced model
        }
        self.model = None
        self.initialize_model()  # Initialisation immédiate

        if not use_llm:
            self.logger.info(f"Mode sans LLM activé. Stratégie de fallback: {llm_fallback_strategy}")

    def initialize_model(self) -> None:
        """Initialise l'architecture du modèle."""
        try:
            # Construction directe via la fonction wrapper, passant toutes les dimensions
            self.model = build_enhanced_hybrid_model(
                tech_input_shape=(self.config["num_technical_features"],),
                llm_embedding_dim=self.config["llm_embedding_dim"],
                mcp_input_dim=self.config["mcp_input_dim"],
                hmm_input_dim=self.config["hmm_input_dim"],  # Pass HMM dim
                instrument_vocab_size=self.config["instrument_vocab_size"],
                instrument_embedding_dim=self.config["instrument_embedding_dim"],
                num_trading_classes=self.config["num_signal_classes"],
                num_market_regime_classes=self.config["num_market_regimes"],
                num_volatility_quantiles=self.config.get("num_volatility_quantiles", 3),  # Use get for optional keys
                num_sl_tp_outputs=self.config.get("num_sl_tp_outputs", 2),  # Use get for optional keys
                # Paramètres pour la gestion des données LLM
                use_llm=self.use_llm,
                llm_fallback_strategy=self.llm_fallback_strategy,
                # active_outputs can be passed here if needed
            )
            self.logger.info("Modèle initialisé avec succès")
            if not self.use_llm:
                self.logger.info(f"Modèle configuré sans LLM. Stratégie de fallback: {self.llm_fallback_strategy}")
            self.logger.info(f"Architecture du modèle:\n{self.get_model_summary()}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")
            raise RuntimeError(f"Échec de l'initialisation: {str(e)}") from e

    def predict(
        self,
        technical_data: np.ndarray,
        llm_embeddings: np.ndarray = None,
        mcp_data: np.ndarray = None,
        hmm_data: np.ndarray = None,
        instrument_data: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:
        """
        Effectue une prédiction complète avec les entrées disponibles.

        Args:
            technical_data: Données techniques (shape: [batch_size, num_technical_features])
            llm_embeddings: Embeddings LLM (shape: [batch_size, llm_embedding_dim]) - Optionnel si use_llm=False
            mcp_data: Données MCP (shape: [batch_size, mcp_input_dim]) - Valeur par défaut si non fournie
            hmm_data: Données HMM (shape: [batch_size, hmm_input_dim]) - Valeur par défaut si non fournie
            instrument_data: IDs d'instrument (shape: [batch_size, 1]) - Valeur par défaut si non fournie

        Returns:
            Dictionnaire contenant toutes les prédictions avec les clés (selon les sorties actives du modèle):
            - 'signal' (shape: [batch_size, 5])
            - 'volatility_quantiles' (shape: [batch_size, 3])
            - 'volatility_regime' (shape: [batch_size, 3])
            - 'market_regime' (shape: [batch_size, num_regime_classes])
            - 'sl_tp' (shape: [batch_size, 2])
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé")

        try:
            batch_size = technical_data.shape[0]

            # Validation de la shape des données techniques (toujours requises)
            if technical_data.shape[1] != self.config["num_technical_features"]:
                raise ValueError(
                    f"Shape technique incorrecte: attendu {self.config['num_technical_features']}, reçu {technical_data.shape[1]}"
                )

            # Génération de données par défaut si non fournies
            if mcp_data is None:
                self.logger.warning("Données MCP non fournies. Utilisation de zéros.")
                mcp_data = np.zeros((batch_size, self.config["mcp_input_dim"]))
            elif mcp_data.shape[1] != self.config["mcp_input_dim"]:
                raise ValueError(
                    f"Shape MCP incorrecte: attendu {self.config['mcp_input_dim']}, reçu {mcp_data.shape[1]}"
                )

            if hmm_data is None:
                self.logger.warning("Données HMM non fournies. Utilisation de zéros.")
                hmm_data = np.zeros((batch_size, self.config["hmm_input_dim"]))
            elif hmm_data.shape[1] != self.config["hmm_input_dim"]:
                raise ValueError(
                    f"Shape HMM incorrecte: attendu {self.config['hmm_input_dim']}, reçu {hmm_data.shape[1]}"
                )

            if instrument_data is None:
                self.logger.warning("Données d'instrument non fournies. Utilisation de 0 (spot).")
                instrument_data = np.zeros((batch_size, 1), dtype=np.int64)
            elif instrument_data.shape[1] != 1:
                raise ValueError(f"Shape Instrument incorrecte: attendu 1, reçu {instrument_data.shape[1]}")

            # Gérer les embeddings LLM selon le mode du modèle
            if self.use_llm:
                if llm_embeddings is None:
                    raise ValueError("Embeddings LLM requis mais non fournis (use_llm=True)")
                if llm_embeddings.shape[1] != self.config["llm_embedding_dim"]:
                    raise ValueError(
                        f"Shape LLM incorrecte: attendu {self.config['llm_embedding_dim']}, reçu {llm_embeddings.shape[1]}"
                    )
            elif llm_embeddings is not None:
                self.logger.warning("Embeddings LLM fournis mais non utilisés (use_llm=False)")

            # Préparation du dictionnaire d'entrées
            input_dict = {
                "technical_input": technical_data,
                "mcp_input": mcp_data,
                "hmm_input": hmm_data,
                "instrument_input": instrument_data,
            }

            # Ajouter les embeddings LLM si nécessaire
            if self.use_llm:
                input_dict["llm_input"] = llm_embeddings
            predictions = self.model.predict(input_dict)

            # Mapping des sorties selon les noms définis dans le modèle
            # Le modèle retourne un dictionnaire si les sorties sont nommées, sinon une liste
            if isinstance(predictions, dict):
                output_dict = predictions  # Déjà un dictionnaire
            else:
                # Si c'est une liste, mapper manuellement basé sur l'ordre attendu
                # Cet ordre dépend de la définition dans build_enhanced_hybrid_model
                # Supposons l'ordre: signal, volatility_quantiles, market_regime, sl_tp
                output_names = ["signal", "volatility_quantiles", "market_regime", "sl_tp"]  # Ajuster si l'ordre change
                output_dict = {name: predictions[i] for i, name in enumerate(output_names) if i < len(predictions)}

            self.logger.info(f"Prédiction réussie pour {technical_data.shape[0]} échantillons")
            return output_dict
            """
            # Ancien mapping basé sur une liste de sortie fixe
            return {
                'signal': predictions[0], # Index basé sur l'ordre de sortie du modèle
                'volatility_quantiles': predictions[1],
                #'volatility_regime': predictions[2], # Probablement obsolète
                'market_regime': predictions[2], # Ajuster l'index si l'ordre change
                'sl_tp': predictions[3] # Ajuster l'index si l'ordre change
            }
            """

        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise  # Relance l'exception originale

    def save_weights(self, filepath: str) -> None:
        """Sauvegarde les poids du modèle."""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé")
        self.model.save_weights(filepath)
        self.logger.info(f"Poids du modèle sauvegardés dans {filepath}")

    def load_weights(self, filepath: str) -> None:
        """Charge les poids du modèle."""
        if self.model is None:
            self.initialize_model()
        self.model.load_weights(filepath)
        self.logger.info(f"Poids du modèle chargés depuis {filepath}")

    def get_model_summary(self) -> str:
        """Retourne un résumé textuel de l'architecture du modèle."""
        if self.model is None:
            return "Modèle non initialisé"

        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)

    def prepare_for_inference(self) -> None:
        """Prépare le modèle pour l'inférence (optimisations)."""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé")

        # Optimisations pour l'inférence
        self.model.compile(optimizer="adam")  # Recompilation légère
        self.logger.info("Modèle optimisé pour l'inférence")
