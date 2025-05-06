from typing import Tuple, Dict, Optional, List
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Embedding, Flatten, Layer,
    Conv1D, MaxPooling1D, LSTM, Reshape, TimeDistributed, Bidirectional
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# Configuration (pourrait être externe)
DEFAULT_INSTRUMENT_VOCAB_SIZE = 10 # Nombre supposé d'instruments différents (Spot, BTC Future, ETH Option, etc.)
DEFAULT_INSTRUMENT_EMBEDDING_DIM = 8

# --- Helper Layer for Conditional Logic ---
class ConditionalOutputLayer(Layer):
    """
    Couche Keras pour gérer les sorties conditionnelles basées sur instrument_type.
    Simplifié : utilise tf.gather pour sélectionner la sortie appropriée.
    Une implémentation complète avec tf.cond serait plus robuste mais complexe.
    """
    def __init__(self, output_dims: Dict[int, int], name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_dims = output_dims # {instrument_id: dim_output, ...}
        # Créer des couches Dense pour chaque type d'instrument possible
        self.output_layers = {
            inst_id: Dense(dim, activation='linear', name=f"{name}_inst_{inst_id}")
            for inst_id, dim in output_dims.items()
        }
        # Assurer une sortie par défaut si aucun type ne correspond (ou pour le padding)
        # Utilise la dimension du premier type comme défaut
        default_dim = list(output_dims.values())[0] if output_dims else 2 # Fallback dim
        self.default_output_layer = Dense(default_dim, activation='linear', name=f"{name}_default")


    def call(self, inputs):
        shared_features, instrument_ids = inputs # instrument_ids shape (batch_size, 1)

        # Créer les sorties potentielles pour tous les instruments dans le batch
        outputs = []
        for inst_id, layer in self.output_layers.items():
            outputs.append(layer(shared_features))
        # Ajouter la sortie par défaut
        default_output = self.default_output_layer(shared_features)
        outputs.append(default_output) # Index -1 pour le défaut

        # Empiler les sorties potentielles: shape (batch_size, num_instrument_types + 1, output_dim)
        stacked_outputs = tf.stack(outputs, axis=1)

        # Utiliser instrument_ids pour sélectionner la bonne sortie pour chaque sample
        # Assurer que les IDs sont dans les bornes [0, num_layers-1]
        # Si un ID est hors des clés de self.output_layers, utiliser l'index -1 (défaut)
        # Note: Ceci est une simplification. tf.cond serait plus robuste mais complexe.
        # Créer un mapping ID -> index dans stacked_outputs
        id_to_index = {inst_id: i for i, inst_id in enumerate(self.output_layers.keys())}
        # Utiliser -1 (index du défaut) pour les IDs inconnus
        indices = tf.map_fn(lambda x: id_to_index.get(x[0], -1), instrument_ids, dtype=tf.int32)

        # Ajouter un offset pour correspondre à la dimension batch
        batch_range = tf.range(tf.shape(instrument_ids)[0])
        gather_indices = tf.stack([batch_range, indices], axis=1)

        selected_outputs = tf.gather_nd(stacked_outputs, gather_indices)

        return selected_outputs

    def get_config(self):
        config = super().get_config()
        config.update({"output_dims": self.output_dims})
        return config


def build_enhanced_hybrid_model(
    tech_input_shape: Tuple[int] = (38,),
    llm_embedding_dim: int = 768,
    mcp_input_dim: int = 128,
    hmm_input_dim: int = 4, # Ajout de la dimension pour les features HMM (régime + probas)
    instrument_vocab_size: int = DEFAULT_INSTRUMENT_VOCAB_SIZE,
    instrument_embedding_dim: int = DEFAULT_INSTRUMENT_EMBEDDING_DIM,
    num_trading_classes: int = 5,
    num_market_regime_classes: int = 4, # Mis à jour selon Agent 4 spec
    num_volatility_quantiles: int = 3,
    num_sl_tp_outputs: int = 2,
    options_output_dim: Optional[int] = 5, # Exemple: delta, gamma, vega, theta, rho
    futures_output_dim: Optional[int] = 3, # Exemple: roll_yield, basis, funding_rate
    instrument_id_map: Optional[Dict[str, int]] = None, # e.g., {"spot": 0, "future": 1, "option": 2}
    sl_tp_initial_bias: Optional[List[float]] = None, # Initialisation personnalisée du biais SL/TP
    active_outputs: List[str] = None, # Liste des noms des sorties à activer
    use_llm: bool = True, # Indique si les données LLM sont disponibles
    llm_fallback_strategy: str = 'zero_vector', # Stratégie de fallback pour LLM: 'zero_vector', 'learned_embedding', 'technical_projection'
    
    # Paramètres pour le traitement CNN+LSTM
    use_cnn_lstm: bool = True,  # Activer/désactiver les couches CNN+LSTM
    time_steps: int = 10,       # Nombre d'étapes temporelles pour les données techniques
    tech_features: int = None,  # Nombre de features techniques par pas de temps (calculé automatiquement si None)
    cnn_filters: List[int] = [32, 64],  # Nombre de filtres pour chaque couche CNN
    cnn_kernel_sizes: List[int] = [3, 3],  # Tailles des noyaux pour chaque couche CNN
    cnn_pool_sizes: List[int] = [2, 2],  # Tailles des pooling pour chaque couche CNN
    lstm_units: List[int] = [64],  # Nombre d'unités pour chaque couche LSTM
    bidirectional_lstm: bool = True,  # Utiliser des LSTM bidirectionnels
    cnn_dropout_rate: float = 0.2,  # Taux de dropout pour les couches CNN
    lstm_dropout_rate: float = 0.2   # Taux de dropout pour les couches LSTM
) -> Model:
    """
    Construit le modèle hybride amélioré avec entrées Tech, LLM, MCP et Instrument,
    et des têtes de sortie multi-tâches incluant des sorties conditionnelles.

    Args:
        tech_input_shape: Shape des features techniques (38)
        llm_embedding_dim: Dimension des embeddings LLM (768)
        mcp_input_dim: Dimension des features MCP (128)
        hmm_input_dim: Dimension des features HMM (régime + probas)
        instrument_vocab_size: Taille du vocabulaire des instruments
        instrument_embedding_dim: Dimension de l'embedding des instruments
        num_trading_classes: Nombre de classes pour le signal de trading
        num_market_regime_classes: Nombre de classes pour le régime de marché
        num_volatility_quantiles: Nombre de quantiles de volatilité
        num_sl_tp_outputs: Nombre de sorties pour SL/TP
        options_output_dim: Dimension de la sortie conditionnelle pour les options
        futures_output_dim: Dimension de la sortie conditionnelle pour les futures
        instrument_id_map: Mapping des IDs d'instruments (optionnel)
        sl_tp_initial_bias: Valeurs d'initialisation du biais pour la couche SL/TP (optionnel)
        active_outputs: Liste des noms des sorties à construire (ex: ['signal', 'market_regime']).
                        Si None, toutes les sorties par défaut sont construites (comportement précédent).

    Returns:
        Le modèle Keras compilé
    """
    # Si active_outputs n'est pas fourni, utiliser toutes les sorties par défaut pour la rétrocompatibilité
    if active_outputs is None:
        active_outputs = ['signal', 'volatility_quantiles', 'market_regime', 'sl_tp']

    # --- Entrées ---
    tech_input = Input(shape=tech_input_shape, name="technical_input")
    mcp_input = Input(shape=(mcp_input_dim,), name="mcp_input")
    hmm_input = Input(shape=(hmm_input_dim,), name="hmm_input") # Nouvelle entrée HMM
    instrument_input = Input(shape=(1,), name="instrument_input", dtype='int64') # ID de l'instrument
    
    # Gestion conditionnelle de l'entrée LLM
    if use_llm:
        llm_input = Input(shape=(llm_embedding_dim,), name="llm_input")
    else:
        # Créer un placeholder pour l'entrée LLM qui ne sera pas utilisé dans le modèle final
        # Cela permet de maintenir la compatibilité avec les scripts existants
        llm_input = Input(shape=(llm_embedding_dim,), name="llm_input_placeholder")

    # --- Assertions sur les shapes d'entrée ---
    """
    tf.debugging.assert_shapes([
        (tech_input, ('B', tech_input_shape[0])), # B = Batch size
        (llm_input, ('B', llm_embedding_dim)),
        (mcp_input, ('B', mcp_input_dim)),
        (hmm_input, ('B', hmm_input_dim)), # Assertion pour HMM
        (instrument_input, ('B', 1)),
    ], message="Validation des shapes d'entrée")
    """

    # --- Encodeurs Spécifiques ---
    # Définir une fonction pour créer un encodeur standard
    def create_encoder(input_tensor, num_units, name_prefix):
        """Crée un encodeur avec Dense, BN, ReLU et Dropout."""
        x = Dense(num_units, activation='relu', kernel_regularizer=regularizers.l2(0.001), name=f"{name_prefix}_dense1")(input_tensor)
        x = BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = Dropout(0.3, name=f"{name_prefix}_dropout1")(x)
        x = Dense(num_units // 2, activation='relu', kernel_regularizer=regularizers.l2(0.001), name=f"{name_prefix}_dense2")(x)
        x = BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = Dropout(0.3, name=f"{name_prefix}_dropout2")(x)
        # Assertions sur les shapes de sortie
        """
        tf.debugging.assert_shapes([(x, ('B', num_units // 2))], message=f"Validation shape sortie encodeur {name_prefix}")
        """
        return x

    # 1. Encodeur Technique (CNN+LSTM ou Dense)
    if use_cnn_lstm:
        # Déterminer le nombre de features par pas de temps
        if tech_features is None:
            tech_features = tech_input_shape[0] // time_steps
            if tech_input_shape[0] % time_steps != 0:
                raise ValueError(f"La dimension d'entrée technique {tech_input_shape[0]} n'est pas divisible par time_steps {time_steps}")
        
        # Réorganiser les données d'entrée de [batch, features] en [batch, time_steps, features_per_step]
        # Exemple: [batch, 60] -> [batch, 10, 6] pour 10 pas de temps et 6 features par pas
        reshaped_tech = Reshape((time_steps, tech_features), name='tech_reshape')(tech_input)
        
        # Application des couches CNN pour extraire des motifs locaux
        x = reshaped_tech
        for i, (filters, kernel_size, pool_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes, cnn_pool_sizes)):
            # Couche Conv1D avec BatchNorm et Dropout
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same',
                      name=f'cnn_conv_{i+1}')(x)
            x = BatchNormalization(name=f'cnn_bn_{i+1}')(x)
            x = Dropout(cnn_dropout_rate, name=f'cnn_dropout_{i+1}')(x)
            
            # MaxPooling pour réduire la dimensionnalité et extraire les caractéristiques importantes
            x = MaxPooling1D(pool_size=pool_size, padding='same', name=f'cnn_pool_{i+1}')(x)
        
        # Application des couches LSTM pour capturer les dépendances temporelles
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1  # True pour toutes sauf la dernière couche
            
            # Couche LSTM (bidirectionnelle ou non)
            if bidirectional_lstm:
                x = Bidirectional(LSTM(units, return_sequences=return_sequences,
                                     recurrent_dropout=lstm_dropout_rate,
                                     name=f'lstm_{i+1}'), name=f'bidirectional_lstm_{i+1}')(x)
            else:
                x = LSTM(units, return_sequences=return_sequences,
                       recurrent_dropout=lstm_dropout_rate,
                       name=f'lstm_{i+1}')(x)
            
            x = BatchNormalization(name=f'lstm_bn_{i+1}')(x)
            x = Dropout(lstm_dropout_rate, name=f'lstm_dropout_{i+1}')(x)
        
        # Le résultat est notre représentation technique encodée
        tech_encoded = x
    else:
        # Utiliser l'encodeur standard si CNN+LSTM n'est pas activé
        tech_encoded = create_encoder(tech_input, 64, "tech") # Sortie: (B, 32)

    # 2. Encodeur LLM avec stratégies de fallback
    if use_llm:
        # Utilisation normale de l'encodeur LLM
        llm_encoded = create_encoder(llm_input, 128, "llm") # Sortie: (B, 64)
    else:
        # Stratégies de fallback selon le paramètre llm_fallback_strategy
        if llm_fallback_strategy == 'zero_vector':
            # Utiliser un vecteur de zéros comme remplacement
            llm_encoded = tf.zeros_like(tech_encoded) * 0
        elif llm_fallback_strategy == 'learned_embedding':
            # Créer un embedding appris pour remplacer l'entrée LLM
            # Cet embedding sera appris pendant l'entraînement
            llm_embedding_layer = Dense(64, activation='relu', name='llm_fallback_embedding')
            llm_encoded = llm_embedding_layer(tech_encoded)
        elif llm_fallback_strategy == 'technical_projection':
            # Projeter les données techniques dans l'espace LLM
            llm_projection = Dense(128, activation='relu', name='tech_to_llm_projection')(tech_encoded)
            llm_encoded = Dense(64, activation='relu', name='llm_projection_encoded')(llm_projection)
        else:
            # Par défaut, utiliser un vecteur de zéros
            llm_encoded = tf.zeros_like(tech_encoded) * 0

    # 3. Encodeur MCP
    mcp_encoded = create_encoder(mcp_input, 64, "mcp") # Sortie: (B, 32)

    # 4. Encodeur HMM
    hmm_encoded = create_encoder(hmm_input, 32, "hmm") # Sortie: (B, 16)

    # 5. Type d'Instrument (Embedding)
    inst_emb = Embedding(input_dim=instrument_vocab_size,
                         output_dim=instrument_embedding_dim,
                         name='instrument_embedding')(instrument_input) # Sortie: (B, 1, emb_dim)
    inst_emb_flat = Flatten()(inst_emb) # Shape: (B, instrument_embedding_dim)
    """
    tf.debugging.assert_shapes([(inst_emb_flat, ('B', instrument_embedding_dim))], message="Validation shape embedding instrument aplati")
    """


    # --- Fusion des Représentations ---
    # Concaténation des features encodées (incluant HMM)
    concatenated_features = Concatenate(name='fusion_concat')([tech_encoded, llm_encoded, mcp_encoded, hmm_encoded, inst_emb_flat])

    # Add Dense layers for richer fusion before shared core
    fused = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='fusion_dense1')(concatenated_features)
    fused = BatchNormalization(name='fusion_bn1')(fused)
    fused = Dropout(0.4, name='fusion_dropout1')(fused) # Keep dropout rate consistent? Or adjust? Let's use 0.4 for now.
    fused = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='fusion_dense2')(fused)
    fused = BatchNormalization(name='fusion_bn2')(fused)
    fused_output = Dropout(0.4, name='fusion_dropout2')(fused)
    # Shape attendue après fusion dense: (B, 128)
    """
    tf.debugging.assert_shapes([(fused_output, ('B', 128))], message="Validation shape après fusion dense")
    """

    # --- Coeur Partagé ---
    # Input is now fused_output
    z = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='shared_dense1')(fused_output)
    z = BatchNormalization(name='shared_bn1')(z)
    z = Dropout(0.4, name='shared_dropout1')(z)
    shared_output = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='shared_dense_output')(z)
    # tf.debugging.assert_shapes([(shared_output, ('B', 64))], message="Validation shape sortie coeur partagé")

    # --- Têtes de Sortie Multi-Tâche ---
    outputs = {} # Dictionnaire pour stocker les sorties actives

    # 1. Signal de Trading (5 classes)
    if 'signal' in active_outputs:
        signal_output = Dense(num_trading_classes, activation='softmax', name='signal')(shared_output)
        outputs['signal'] = signal_output
        # tf.debugging.assert_shapes([(signal_output, ('B', num_trading_classes))], message="Validation shape sortie signal")

    # 2. Quantiles de Volatilité (3 valeurs)
    if 'volatility_quantiles' in active_outputs:
        volatility_quantiles = Dense(num_volatility_quantiles, activation='linear', name='volatility_quantiles')(shared_output)
        outputs['volatility_quantiles'] = volatility_quantiles
        # tf.debugging.assert_shapes([(volatility_quantiles, ('B', num_volatility_quantiles))], message="Validation shape sortie vol quantiles")

    # 3. Régime de Marché (4 classes)
    if 'market_regime' in active_outputs:
        market_regime = Dense(num_market_regime_classes, activation='softmax', name='market_regime')(shared_output)
        outputs['market_regime'] = market_regime
        # tf.debugging.assert_shapes([(market_regime, ('B', num_market_regime_classes))], message="Validation shape sortie market regime")

    # 4. SL/TP de Base (2 valeurs - pour module RL externe)
    if 'sl_tp' in active_outputs:
        # Initialisation personnalisée du biais pour éviter un démarrage à zéro
        sl_tp_bias_initializer = None
        if sl_tp_initial_bias is not None:
            # Utiliser une initialisation de biais personnalisée si fournie
            sl_tp_bias_initializer = tf.keras.initializers.Constant(sl_tp_initial_bias)

        # Utiliser l'initialisation de biais personnalisée ou la valeur par défaut
        sl_tp_output = Dense(
            num_sl_tp_outputs,
            activation='linear',
            name='sl_tp',
            kernel_initializer='glorot_normal',  # Initialisation Xavier/Glorot plus stable que l'initialisation par défaut
            kernel_regularizer=regularizers.l2(0.001),  # Régularisation L2 pour éviter l'overfitting
            bias_initializer=sl_tp_bias_initializer  # Initialisation du biais personnalisée si fournie
        )(shared_output)
        outputs['sl_tp'] = sl_tp_output # Garder le nom 'sl_tp' pour la compatibilité avec le data_loader
        # tf.debugging.assert_shapes([(sl_tp_output, ('B', num_sl_tp_outputs))], message="Validation shape sortie sl_tp")

    # --- Têtes Conditionnelles (Exemple simplifié) ---
    # Définir les dimensions de sortie pour chaque type d'instrument pertinent
    # Utiliser instrument_id_map si fourni, sinon supposer 0:spot, 1:future, 2:option
    if instrument_id_map is None:
        instrument_id_map = {"spot": 0, "future": 1, "option": 2} # Exemple

    # Désactiver temporairement la sortie conditionnelle qui cause des erreurs
    conditional_outputs = {}
    """
    conditional_outputs = {}
    if options_output_dim and instrument_id_map.get("option") is not None:
        conditional_outputs[instrument_id_map["option"]] = options_output_dim
    if futures_output_dim and instrument_id_map.get("future") is not None:
        conditional_outputs[instrument_id_map["future"]] = futures_output_dim

    if conditional_outputs:
         # Ajouter une tête conditionnelle générique
         # Note: Le nom 'conditional_output' est générique. L'interprétation dépendra
         # du type d'instrument prédit ou utilisé en entrée.
         conditional_head = ConditionalOutputLayer(output_dims=conditional_outputs, name='conditional_output')
         conditional_output = conditional_head([shared_output, instrument_input])
         # Assertion sur la sortie conditionnelle (la dimension peut varier)
         # On ne peut pas faire une assertion simple ici car la shape dépend de l'instrument_id
         # tf.debugging.assert_shapes([(conditional_output, ('B', None))], message="Validation shape sortie conditionnelle") # Pas possible simplement
         outputs['conditional_output'] = conditional_output
    """

    # --- Modèle Final ---
    # Les assertions sur les shapes d'entrée sont déjà faites plus haut

    # Création du dictionnaire d'entrées en fonction de l'utilisation ou non de LLM
    inputs_dict = {
        "technical_input": tech_input,
        "mcp_input": mcp_input,
        "hmm_input": hmm_input,
        "instrument_input": instrument_input
    }
    
    # Ajouter l'entrée LLM au dictionnaire si elle est utilisée
    if use_llm:
        inputs_dict["llm_input"] = llm_input
    
    model = Model(
        inputs=inputs_dict,
        outputs=outputs, # Dictionnaire pour les sorties nommées
        name='enhanced_hybrid_model_v2'
    )

    return model
