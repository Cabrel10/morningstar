import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import telegram  # Ajout de l'import manquant

# Import des fonctions et variables à tester depuis telegram_bot
# Assurez-vous que telegram_bot.py est dans le PYTHONPATH ou accessible
# Pour les tests, il est préférable de ne pas initialiser le vrai bot telegram.Bot
# Nous allons mocker cela.

# Mocker les variables globales avant l'import pour éviter l'initialisation réelle du bot
with patch("telegram_bot.TELEGRAM_BOT_TOKEN", "FAKE_TOKEN"), patch(
    "telegram_bot.TELEGRAM_CHAT_ID", "FAKE_CHAT_ID"
), patch("telegram.Bot", MagicMock()) as mock_bot_class:

    # Créez une instance mock pour bot = telegram.Bot(token=...)
    mock_bot_instance = MagicMock()
    mock_bot_class.return_value = mock_bot_instance

    # Maintenant, importez les éléments de telegram_bot
    from telegram_bot import notify_trade_sync, send_telegram_message, bot as telegram_bot_actual_bot_instance

# Rétablir le bot original après l'import pour d'autres tests potentiels ou si le module est utilisé ailleurs
# Cependant, pour les tests unitaires, il est généralement préférable de mocker au niveau de chaque test.


@pytest.fixture
def mock_asyncio_run():
    with patch("asyncio.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_send_telegram_message_func():
    # Mocker la fonction send_telegram_message au sein du module telegram_bot
    with patch("telegram_bot.send_telegram_message", new_callable=AsyncMock) as mock_send:
        yield mock_send


@pytest.fixture(autouse=True)
def mock_bot_global_instance():
    """Assure que l'instance globale `bot` dans telegram_bot.py est un mock."""
    with patch("telegram_bot.bot", MagicMock(spec=telegram.Bot)) as mock_global_bot:
        # Configurer le mock pour send_message s'il est appelé directement sur l'instance globale
        mock_global_bot.send_message = AsyncMock()
        yield mock_global_bot


def test_notify_trade_sync_calls_send_message(mock_send_telegram_message_func):
    """
    Teste si notify_trade_sync appelle send_telegram_message (via asyncio.run)
    avec le message formaté correctement.
    """
    signal = "TEST_BUY"
    price = 12345.67
    reasoning = "Test reasoning."
    chat_id = "TEST_CHAT_ID"

    expected_message = (
        f"*Trade Alert - Morningstar*\n\n*Signal:* {signal}\n*Price:* ${price:.2f}\n*Reasoning:* {reasoning}"
    )

    # Mocker asyncio.run pour vérifier qu'il est appelé avec la bonne coroutine.
    # La coroutine send_telegram_message est déjà testée séparément (et mockée ici par mock_send_telegram_message_func).
    with patch("asyncio.run") as mock_asyncio_run_call:
        notify_trade_sync(signal, price, reasoning, chat_id)

        # Vérifier que asyncio.run a été appelé une fois.
        mock_asyncio_run_call.assert_called_once()

        # Vérifier que la coroutine passée à asyncio.run est bien un appel à
        # la fonction mockée mock_send_telegram_message_func avec les bons arguments.
        # Pour cela, on récupère l'objet coroutine passé à asyncio.run
        # Note: L'objet coroutine lui-même ne peut pas être directement comparé facilement.
        # On se fie au fait que mock_send_telegram_message_func sera appelé si la coroutine est correcte.
        # Et que mock_send_telegram_message_func est déjà testé pour son comportement.

        # Pour s'assurer que la coroutine correcte a été passée à asyncio.run,
        # on peut vérifier que mock_send_telegram_message_func a été appelé
        # (ce qui se produirait si asyncio.run exécutait la coroutine).
        # Comme nous ne voulons pas exécuter la coroutine ici (juste vérifier l'appel à run),
        # nous allons plutôt vérifier que mock_send_telegram_message_func a été appelé
        # par le wrapper notify_trade_sync.
        # Le test précédent avec side_effect=dummy_async_run_target était plus direct pour cela.
        # Pour éviter le warning, nous allons juste vérifier l'appel à asyncio.run
        # et supposer que si la bonne coroutine est passée, mock_send_telegram_message_func
        # serait appelé avec les bons args (ce qui est testé par son propre mock).

        # Alternative: Si on veut vérifier les arguments de la coroutine passée à asyncio.run
        # sans l'exécuter, c'est plus complexe.
        # Pour ce test, nous allons nous contenter de vérifier que asyncio.run est appelé.
        # Le test de l'appel à mock_send_telegram_message_func est déjà fait par la fixture.
        # Le but de ce test est de s'assurer que notify_trade_sync utilise asyncio.run.

        # Pour ce test spécifique, nous allons nous assurer que la fonction mockée
        # send_telegram_message (qui est mock_send_telegram_message_func) est appelée
        # par notify_trade_sync. La manière dont asyncio.run est gérée dans notify_trade_sync
        # (try/except RuntimeError) rend le test direct de l'argument de asyncio.run un peu plus délicat
        # sans exécuter réellement la coroutine.
        # Le test original avec dummy_async_run_target était correct pour vérifier l'effet final.
        # Le warning est un effet secondaire de ce mock.
        # Pour l'instant, nous allons garder la logique qui vérifie l'appel à la fonction mockée.
        # Si le warning persiste, il faudra peut-être une approche plus sophistiquée pour mocker asyncio.run.

        # Pour ce test, nous allons mocker asyncio.run et vérifier que send_telegram_message
        # (qui est mock_send_telegram_message_func) est appelé.
        # Cela signifie que nous devons permettre à la logique dans notify_trade_sync de l'appeler.

        # On revient à la logique qui exécute la coroutine pour vérifier l'appel interne.
    # Pour ce test, nous voulons vérifier que le chemin principal de notify_trade_sync
    # (le bloc try) appelle send_telegram_message.
    # Nous allons mocker asyncio.run pour qu'il exécute la coroutine directement
    # et nous allons aussi mocker send_telegram_message pour vérifier ses arguments.

    # mock_send_telegram_message_func est une fixture qui mocke telegram_bot.send_telegram_message (AsyncMock)

    # Pour ce test, nous voulons simuler le cas où la première tentative asyncio.run réussit.
    # Nous allons donc patcher asyncio.run pour qu'il exécute la coroutine passée
    # et nous allons vérifier que mock_send_telegram_message_func est appelé une seule fois.

    async def run_coro_once(coro):
        # Ceci exécute la coroutine (qui est mock_send_telegram_message_func)
        await coro
        return "Mocked Run Result"  # La valeur de retour n'importe pas ici

    # Patch asyncio.run pour qu'il utilise notre implémentation ci-dessus.
    # Cela devrait empêcher la RuntimeError et donc le second appel dans le bloc except.
    with patch("asyncio.run", side_effect=run_coro_once) as mock_asyncio_run:
        notify_trade_sync(signal, price, reasoning, chat_id)

        # Vérifie que asyncio.run a été appelé une fois.
        mock_asyncio_run.assert_called_once()
        # Vérifie que la fonction mockée send_telegram_message a été appelée une fois avec les bons arguments.
        mock_send_telegram_message_func.assert_called_once_with(chat_id, expected_message)


@pytest.mark.asyncio
async def test_send_telegram_message_success(mock_bot_global_instance):
    """Teste l'envoi réussi d'un message."""
    chat_id = "123"
    message = "Hello Test"

    # Configurer le mock de bot.send_message pour qu'il ne lève pas d'erreur
    mock_bot_global_instance.send_message = AsyncMock()  # S'assurer que c'est un AsyncMock

    # Remplacer temporairement le bot global dans telegram_bot par notre mock_bot_global_instance
    # si ce n'est pas déjà fait par autouse=True (ce qui devrait être le cas)
    # telegram_bot_actual_bot_instance = mock_bot_global_instance # Redondant si autouse

    result = await send_telegram_message(chat_id, message)

    assert result is True
    mock_bot_global_instance.send_message.assert_awaited_once_with(chat_id=chat_id, text=message, parse_mode="Markdown")


@pytest.mark.asyncio
async def test_send_telegram_message_bot_not_initialized():
    """Teste le cas où le bot n'est pas initialisé."""
    with patch("telegram_bot.bot", None):  # Simuler bot = None
        result = await send_telegram_message("123", "Test")
    assert result is False


@pytest.mark.asyncio
async def test_send_telegram_message_invalid_chat_id(mock_bot_global_instance):
    """Teste le cas d'un chat_id invalide (placeholder)."""
    # Assurer que le bot est initialisé pour ce test
    with patch("telegram_bot.bot", mock_bot_global_instance):
        result = await send_telegram_message("YOUR_TELEGRAM_CHAT_ID", "Test")
    assert result is False
    mock_bot_global_instance.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_telegram_message_api_error(mock_bot_global_instance):
    """Teste la gestion d'une exception lors de l'envoi."""
    mock_bot_global_instance.send_message = AsyncMock(side_effect=telegram.error.TelegramError("Test API Error"))

    with patch("telegram_bot.bot", mock_bot_global_instance):
        result = await send_telegram_message("123", "Test")

    assert result is False
    mock_bot_global_instance.send_message.assert_awaited_once()


# Plus de tests pour les handlers de commandes et le décorateur @restricted suivront.
# Pour l'instant, nous nous concentrons sur les fonctions de notification.

# --- Tests pour les Command Handlers (structure de base) ---


# Mocker Update et Context pour les tests de handlers
@pytest.fixture
def mock_update():
    update = MagicMock(spec=telegram.Update)
    update.effective_user = MagicMock(spec=telegram.User)
    update.effective_user.id = 12345  # User ID factice
    update.effective_user.mention_html = MagicMock(return_value="<a href='tg://user?id=12345'>Test User</a>")
    update.effective_chat = MagicMock(spec=telegram.Chat)
    update.effective_chat.id = 12345  # Chat ID factice pour les tests
    update.message = MagicMock(spec=telegram.Message)
    update.message.reply_html = AsyncMock()
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_context():
    context = MagicMock(spec=telegram.ext.ContextTypes.DEFAULT_TYPE)
    context.args = []
    # context.bot = MagicMock(spec=telegram.Bot) # Déjà mocké globalement par mock_bot_global_instance
    return context


# Importer les handlers après les mocks globaux initiaux
with patch("telegram_bot.TELEGRAM_BOT_TOKEN", "FAKE_TOKEN"), patch(
    "telegram_bot.TELEGRAM_CHAT_ID", "FAKE_CHAT_ID"
), patch("telegram.Bot", MagicMock()):
    from telegram_bot import (
        start_command,
        help_command,
        status_command,
        explain_command,
        override_command,
        restricted,
        ALLOWED_CHAT_IDS,
    )


@pytest.mark.asyncio
async def test_start_command(mock_update, mock_context, mock_bot_global_instance):
    # S'assurer que ALLOWED_CHAT_IDS inclut l'ID de l'utilisateur pour ce test
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await start_command(mock_update, mock_context)
    mock_update.message.reply_html.assert_awaited_once()
    # Vérifier une partie du message si nécessaire


@pytest.mark.asyncio
async def test_help_command(mock_update, mock_context, mock_bot_global_instance):
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await help_command(mock_update, mock_context)
    mock_update.message.reply_text.assert_awaited_once()
    # Vérifier le contenu du message d'aide


@pytest.mark.asyncio
async def test_status_command(mock_update, mock_context, mock_bot_global_instance):
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await status_command(mock_update, mock_context)
    mock_update.message.reply_text.assert_awaited_once()
    # Vérifier que le message contient des éléments attendus (ex: "Statut Morningstar Bot")
    # Un test plus spécifique pour le contenu du statut est ci-dessous.


@pytest.mark.asyncio
@patch("telegram_bot.Path")  # Mock pathlib.Path
@patch("builtins.open", new_callable=MagicMock)  # Mock open
async def test_status_command_with_pnl(
    mock_open_file, mock_path_class, mock_update, mock_context, mock_bot_global_instance
):
    """Teste que la commande /status affiche correctement le PnL depuis live_status.json."""

    # Configurer le mock de Path
    mock_status_file_instance = MagicMock()
    mock_status_file_instance.exists.return_value = True
    mock_status_file_instance.resolve.return_value = (
        "/fake/path/to/live_status.json"  # Pour le message d'erreur si besoin
    )
    mock_path_class.return_value = mock_status_file_instance

    # Simuler le contenu du fichier live_status.json
    fake_status_data = {
        "timestamp": 1672531200,  # 2023-01-01 00:00:00 UTC
        "symbol": "BTC/USDT",
        "position_side": "long",
        "entry_price": 30000.0,
        "current_position_size": 0.1,
        "last_known_balance": {"USDT": {"free": 900.0, "total": 1200.0}},
        "trading_active": True,
        "consecutive_errors": 0,
        "current_pnl": 150.75,  # Le PnL à vérifier
    }
    # Configurer le mock de open().read() pour retourner les données JSON simulées
    mock_file_content = MagicMock()
    mock_file_content.read.return_value = json.dumps(fake_status_data)  # Importer json

    # open() est un context manager, donc __enter__ doit retourner le mock de fichier
    mock_open_file.return_value.__enter__.return_value = mock_file_content

    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await status_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_awaited_once()
    call_args = mock_update.message.reply_text.call_args[0][0]  # Le premier argument textuel

    assert "Statut Morningstar Bot" in call_args
    assert "PnL Cumulé: 150.75 USDT" in call_args  # Vérifier que le PnL est présent et formaté
    assert "Position: long" in call_args
    assert "Symbole: BTC/USDT" in call_args


@pytest.mark.asyncio
@patch("telegram_bot.Path")  # Mock pathlib.Path
async def test_status_command_file_not_found(mock_path_class, mock_update, mock_context, mock_bot_global_instance):
    """Teste le comportement de /status si live_status.json n'est pas trouvé."""
    mock_status_file_instance = MagicMock()
    mock_status_file_instance.exists.return_value = False  # Simuler fichier non trouvé
    mock_status_file_instance.resolve.return_value = "/fake/path/live_status.json"
    mock_path_class.return_value = mock_status_file_instance

    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await status_command(mock_update, mock_context)

    mock_update.message.reply_text.assert_awaited_once()
    call_args = mock_update.message.reply_text.call_args[0][0]
    assert "Fichier de statut `live_status.json` non trouvé" in call_args


@pytest.mark.asyncio
async def test_explain_command_no_args(mock_update, mock_context, mock_bot_global_instance):
    mock_context.args = []
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await explain_command(mock_update, mock_context)
    mock_update.message.reply_text.assert_awaited_with(
        "Veuillez fournir un ID de trade/signal. Usage: `/explain <ID_DU_SIGNAL>`", parse_mode="Markdown"
    )


@pytest.mark.asyncio
async def test_explain_command_with_args(mock_update, mock_context, mock_bot_global_instance):
    mock_context.args = ["SIGNAL_123"]
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await explain_command(mock_update, mock_context)
    mock_update.message.reply_text.assert_awaited_once()
    # Vérifier que la réponse contient "SIGNAL_123"


@pytest.mark.asyncio
async def test_override_command_no_args(mock_update, mock_context, mock_bot_global_instance):
    mock_context.args = []
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await override_command(mock_update, mock_context)
    mock_update.message.reply_text.assert_awaited_with(
        "Usage: `/override <BUY|SELL|HOLD> <SYMBOLE> [PRIX_OPTIONNEL]`", parse_mode="Markdown"
    )


@pytest.mark.asyncio
async def test_override_command_valid_args(mock_update, mock_context, mock_bot_global_instance):
    mock_context.args = ["BUY", "BTC/USDT", "30000"]
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await override_command(mock_update, mock_context)
    mock_update.message.reply_text.assert_awaited_once()
    # Vérifier que la réponse contient "Override reçu: BUY BTC/USDT @ 30000.00"


# --- Tests pour le décorateur @restricted ---
@pytest.mark.asyncio
async def test_restricted_decorator_allowed_user(mock_update, mock_context):
    # Simuler une fonction factice décorée
    mock_inner_func = AsyncMock()
    decorated_func = restricted(mock_inner_func)

    # L'utilisateur est autorisé
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_user.id]):
        await decorated_func(mock_update, mock_context)
    mock_inner_func.assert_awaited_once_with(mock_update, mock_context)
    mock_update.message.reply_text.assert_not_called()  # Ne doit pas envoyer de message d'erreur


@pytest.mark.asyncio
async def test_restricted_decorator_allowed_chat(mock_update, mock_context):
    mock_inner_func = AsyncMock()
    decorated_func = restricted(mock_inner_func)

    # Le chat est autorisé (l'utilisateur pourrait être différent)
    mock_update.effective_user.id = 99999  # Utilisateur différent
    with patch("telegram_bot.ALLOWED_CHAT_IDS", [mock_update.effective_chat.id]):
        await decorated_func(mock_update, mock_context)
    mock_inner_func.assert_awaited_once_with(mock_update, mock_context)
    mock_update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_restricted_decorator_not_allowed(mock_update, mock_context):
    mock_inner_func = AsyncMock()
    decorated_func = restricted(mock_inner_func)

    # L'utilisateur/chat n'est pas autorisé
    mock_update.effective_user.id = 99999
    mock_update.effective_chat.id = 88888
    with patch(
        "telegram_bot.ALLOWED_CHAT_IDS", [11111, 22222]
    ):  # Liste d'autorisation qui n'inclut pas l'utilisateur/chat
        await decorated_func(mock_update, mock_context)

    mock_inner_func.assert_not_awaited()  # La fonction interne ne doit pas être appelée
    mock_update.message.reply_text.assert_awaited_once_with(
        "Désolé, vous n'êtes pas autorisé à utiliser cette commande."
    )


@pytest.mark.asyncio
async def test_restricted_decorator_empty_allowed_list(mock_update, mock_context):
    # Si ALLOWED_CHAT_IDS est vide, le bot devrait répondre (comportement par défaut actuel)
    mock_inner_func = AsyncMock()
    decorated_func = restricted(mock_inner_func)

    with patch("telegram_bot.ALLOWED_CHAT_IDS", []):
        await decorated_func(mock_update, mock_context)
    mock_inner_func.assert_awaited_once_with(mock_update, mock_context)
    mock_update.message.reply_text.assert_not_called()
