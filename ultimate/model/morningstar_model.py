from tensorflow.keras.layers import Input, Dense, LSTM, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from utils.transformer import TransformerBlock
import tensorflow as tf

def build_morningstar_model(n_tech, cot_len, mcp_len):
    """
    Construit le modèle Morningstar monolithique intégrant CNN, LSTM, CoT, RL et GA.
    
    Args:
        n_tech: Dimension des features techniques
        cot_len: Longueur des embeddings Chain-of-Thought
        mcp_len: Dimension des features macro/marché
    
    Returns:
        Un modèle Keras avec 3 têtes de sortie (signal, stop-loss, take-profit)
    """
    # Inputs
    tech_in = Input(shape=(n_tech,), name="technical_input")
    emb_in = Input(shape=(768,), name="embeddings_input")
    cot_in = Input(shape=(cot_len,), name="cot_input")
    inst_in = Input(shape=(1,), name="instrument_input")
    mcp_in = Input(shape=(mcp_len,), name="mcp_input")

    # Fusion initiale
    x = Dense(128, activation="relu")(tech_in)
    x = tf.expand_dims(x, axis=1)
    x = LSTM(64, return_sequences=True)(x)
    
    # Transformer
    for _ in range(2):
        x = TransformerBlock(64, num_heads=4)(x)
    x = GlobalAveragePooling1D()(x)

    # Têtes de sortie
    # Signal (3 classes: ACHAT, VENTE, NEUTRE)
    s = Dense(32, activation="relu")(x)
    s = Dense(3, activation="softmax", name="signal")(s)
    
    # Stop-loss
    sl = Dense(32, activation="relu")(x)
    sl = Dense(1, name="sl_level")(sl)
    
    # Take-profit
    tp = Dense(32, activation="relu")(x)
    tp = Dense(1, name="tp_level")(tp)

    return Model(
        inputs=[tech_in, emb_in, cot_in, inst_in, mcp_in],
        outputs=[s, sl, tp],
        name="Morningstar"
    )
