# Método CASTLE

Los tres notebooks del reto **«De la Emoción a la Ecuación»** de Castle Quant: cómo se construye, paso a paso, un sistema de trading cuantitativo sobre el futuro perpetuo **BTCUSDT** de Binance, desde los datos crudos hasta un bot que opera solo en una cuenta demo.

> **Aviso.** Material educativo. No es asesoría financiera ni una recomendación de inversión. Los resultados son simulaciones sobre datos históricos y no garantizan resultados futuros. El bot solo opera en **Binance Demo Trading**, con dinero ficticio.

## El método, notebook por notebook

| Notebook | Letras | Qué se construye | Dónde se ejecuta |
|---|---|---|---|
| `castle1_captura_alfa.ipynb` | **C · A** | **Captura:** velas de 1 minuto de Binance, agregadas a 1h, 6h y 12h. **Alfa:** un modelo lineal en PyTorch que predice el retorno de la próxima vela. | Google Colab o local |
| `castle2_simulacion_tamano_sesgos.ipynb` | **S · T · L** | **Simulación:** de la predicción a las operaciones, con comisiones. **Tamaño:** constante, compuesto, Kelly, apalancamiento y escenarios con Monte Carlo. **Libre de sesgos:** anticipación, selección y prueba de permutaciones. | Google Colab o local |
| `castle3_ejecucion.ipynb` | **E** | **Ejecución:** el sistema que procesa los precios uno a uno y el bot conectado a Binance Demo, con avisos por Telegram. | Local |

Los tres tienen las salidas guardadas: se pueden leer sin ejecutar nada.

## Lo que vas a encontrar

El modelo se elige en un tramo de **validación** (dentro de muestra), donde la estrategia casi duplica el capital después de comisiones, y se prueba una sola vez en un tramo que nunca vio (noviembre de 2025 a octubre de 2026). El resultado es honesto: **fuera de muestra el modelo pierde y no le gana al azar**. El notebook 2 muestra las dos caras y plantea hipótesis sobre el tipo de mercado en el que funciona; la cuenta demo del notebook 3 sirve para recoger los datos que permiten comprobarlas. El bot es **funcional, no rentable**: el alfa es la pieza que se cambia.

## Cómo ejecutarlo

**En Google Colab (notebooks 1 y 2).** Abre el notebook y pulsa *Entorno de ejecución → Ejecutar todo*. La primera celda descarga este repositorio. El notebook 1 baja unos 100 MB de velas la primera vez (uno o dos minutos).

**En tu computador (los tres notebooks y el bot).** Necesitas Python 3.10 o superior.

```bash
git clone https://github.com/JuliRey2000/metodo_CASTLE_castlequant.com.git
cd metodo_CASTLE_castlequant.com
pip install -r requirements.txt
```

Abre la carpeta en [Visual Studio Code](https://code.visualstudio.com/) con las extensiones *Python* y *Jupyter*, entra a un notebook y pulsa **Run All**. El orden importa: el notebook 1 genera `model_weights.pth` y `BTCUSDT_12h_ohlc.csv`, que usan el 2 y el 3 (el repositorio ya los trae).

## El bot

Cada 12 horas (00:00 y 12:00 UTC; 7 p. m. y 7 a. m. en Colombia), cuando cierra la vela:

1. Lee los cierres de las últimas velas de 12 horas del **mercado real** (datos públicos, sin claves).
2. Calcula los retornos y la predicción del modelo.
3. Si la predicción es positiva queda comprado; si es negativa, vendido. Solo opera cuando la dirección cambia.
4. Envía la orden a la **cuenta demo**, la anota en `bot/registros/` y te avisa por Telegram.

Antes de empezar necesitas una cuenta en **Binance Demo Trading** con futuros y una **clave de API creada dentro de Demo Trading**. El notebook 3 guía los pasos; desde una terminal en la carpeta `metodo_CASTLE_castlequant.com` son estos:

| Comando | Qué hace |
|---|---|
| `python bot/bot.py --dry-run --once` | Una decisión con datos reales y exchange simulado. No necesita claves. |
| `python bot/bot.py --check` | Verifica las claves y muestra la cuenta demo. No envía órdenes. |
| `python bot/bot.py --test-order` | Compra y vende la cantidad mínima, para confirmar que la clave puede operar. |
| `python bot/bot.py --telegram` | Te dice tu `chat_id` y envía un mensaje de prueba. |
| `python bot/bot.py` | Enciende el bot. La terminal debe quedar abierta y el computador sin suspenderse. |
| `python bot/bot.py --close-position` | Cierra la posición abierta. Al detener el bot (`Ctrl + C`) la posición **no** se cierra sola. |

**Claves.** Copia `bot/.env.example` como `bot/.env` y pega ahí tus claves. Ese archivo está excluido de Git. No pegues las claves en notebooks, chats ni capturas de pantalla.

**Configuración** (`bot/config.json`):

| Parámetro | Valor | Significado |
|---|---|---|
| `capital_usdt` | 1000 | Capital que maneja el bot. Si el saldo demo es menor, usa el 95 % del saldo. |
| `leverage` | 1 | Apalancamiento, número entero. El notebook 2 muestra lo que pasa al subirlo. |
| `band` | 0.2 | El bot solo opera si la posición se aleja más de 20 puntos del objetivo; en la práctica, cuando cambia la dirección. |
| `slippage` | 0.0001 | Deslizamiento que se supone en el modo simulado. |
| `close_delay_seconds` | 5 | Segundos que espera tras el cierre de la vela, para que Binance termine de publicarla. |

Si se reinicia el computador, basta con volver a ejecutar el bot: lee la posición de la cuenta demo y continúa sin duplicar órdenes.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `binance.py` | Descarga y guarda en caché las velas de 1 minuto; define las comisiones. |
| `research.py` | Agregación de velas, retornos y rezagos, entrenamiento, métricas, simulación y pruebas de sesgos. |
| `models.py` | El modelo lineal de PyTorch. |
| `model_weights.pth` | El modelo entrenado por el notebook 1. |
| `BTCUSDT_12h_ohlc.csv` | Velas de 12 horas (enero de 2022 a octubre de 2026). |
| `bot/` | El bot: estrategia (`bot.py`), bucle y comandos (`comun.py`), datos en vivo (`market_data.py`), cuenta demo (`binance_demo.py`), órdenes y exchange simulado (`trading_system.py`) y avisos (`avisos.py`). |

## Créditos y licencia

El modelo base y parte de las librerías provienen del curso *Build a Quant Trading System* de **MemLabs** ([repositorio original](https://github.com/memlabs-research/build-a-quant-trading-strategy)), publicado bajo licencia MIT, y de la **Mentoría Quant** de Julián Castillo. La adaptación al Método CASTLE, los datos, la partición con validación, las pruebas de sesgos y el bot son de **Castle Quant**. Todo se distribuye bajo licencia MIT. Ver [`LICENSE`](LICENSE).
