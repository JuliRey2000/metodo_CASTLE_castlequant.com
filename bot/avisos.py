"""
avisos.py - Avisos por Telegram
===============================
Envía cada decisión del bot a tu Telegram. Es opcional: si en el archivo .env
no hay token, el bot funciona igual y solo escribe en pantalla.

Proyecto: Castle Quant · Método CASTLE (E · Ejecución)
"""

import logging
import os
from typing import Optional

import requests

API_URL = "https://api.telegram.org"
log = logging.getLogger("bot")


def _call(token: str, method: str, params: dict) -> Optional[dict]:
    """
    Llama a la API de Telegram. Si falla, lo anota y devuelve None: un aviso que no sale no debe detener al bot.

    El token viaja en la dirección de la petición, así que de un error solo se registra su tipo, nunca su texto.
    """
    try:
        response = requests.post(f"{API_URL}/bot{token}/{method}", data=params, timeout=15)
    except requests.RequestException as e:
        log.warning("Telegram: no se pudo conectar (%s)", type(e).__name__)
        return None
    if response.status_code != 200:
        log.warning("Telegram respondió %d. Revisa TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en el archivo .env", response.status_code)
        return None
    return response.json()


def enabled() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send(text: str) -> bool:
    """Envía un mensaje al chat configurado. Devuelve True si salió."""
    if not enabled():
        return False
    result = _call(os.environ["TELEGRAM_BOT_TOKEN"], "sendMessage", {"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text})
    return result is not None


def find_chat_id() -> Optional[int]:
    """Busca tu chat_id: es el del último chat que le escribió a tu bot."""
    result = _call(os.environ.get("TELEGRAM_BOT_TOKEN", ""), "getUpdates", {})
    chats = [u["message"]["chat"]["id"] for u in (result or {}).get("result", []) if "message" in u]
    return chats[-1] if chats else None


def setup() -> None:
    """Comprueba la configuración de Telegram: muestra el chat_id que falta o envía un mensaje de prueba."""
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        print("Falta TELEGRAM_BOT_TOKEN en bot/.env. Crea tu bot con @BotFather en Telegram y pega ahí el token.")
    elif not os.environ.get("TELEGRAM_CHAT_ID"):
        chat_id = find_chat_id()
        if chat_id is None:
            print("Abre Telegram, escríbele cualquier mensaje a tu bot y vuelve a ejecutar este comando.")
        else:
            print(f"Tu chat_id es {chat_id}. Pégalo en bot/.env como TELEGRAM_CHAT_ID={chat_id} y vuelve a ejecutar este comando.")
    elif send("Bot CASTLE conectado. Aquí vas a recibir cada decisión."):
        print("Mensaje de prueba enviado. Revisa tu Telegram.")
    else:
        print("No se pudo enviar el mensaje de prueba.")
