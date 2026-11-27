# Castle Labs

Los notebooks de práctica de la mentoría **Castle Labs** de Castle Quant. Van encima del Método CASTLE: usan las mismas librerías del repositorio (`research.py`, `binance.py`, `models.py`) y los mismos datos.

> **Aviso.** Material educativo. No es asesoría financiera ni una recomendación de inversión. Los resultados son simulaciones sobre datos históricos, y en el notebook de opciones sobre datos inventados. No garantizan resultados futuros.

## ¿Qué hay acá?

| Notebook | Sesión | Qué vamos a hacer |
|---|---|---|
| `sesion4_practica_ficha_de_alfa.ipynb` | **Sesión 4** · Captura y Alfa | Cambias dos o tres parámetros (mercado, intervalo, rezagos) y sales con tu **ficha de alfa**: variables, pesos, tasa de acierto, EV, Sharpe, caída máxima y EV frente al costo. Tal como viene reproduce el modelo de 12 horas del Notebook 1. |
| `castle_walls_opciones.ipynb` | **Sesión 5** · Variables nuevas | Put walls y call walls: cómo se calcula la exposición gamma, los muros y el gamma flip, cómo se guarda una fila diaria y cómo se prueba contra el azar y contra un placebo. No enseña a operar el nivel. |

Los dos tienen las salidas guardadas, así que se pueden leer sin ejecutar nada. Y los dos corren **sin internet** tal como vienen.

## Cómo abrirlos

**En Google Colab.** Abre el enlace y pulsa *Entorno de ejecución → Ejecutar todo*. La primera celda descarga el repositorio.

- Sesión 4: `https://colab.research.google.com/github/JuliRey2000/metodo_CASTLE_castlequant.com/blob/main/castle_labs/sesion4_practica_ficha_de_alfa.ipynb`
- Sesión 5: `https://colab.research.google.com/github/JuliRey2000/metodo_CASTLE_castlequant.com/blob/main/castle_labs/castle_walls_opciones.ipynb`

**En tu computador.** Sigue los pasos del `README.md` principal (clonar e instalar `requirements.txt`), abre la carpeta en Visual Studio Code y pulsa **Run All**. Funcionan igual si abres el notebook desde la raíz del repositorio o desde `castle_labs/`.

## Para que lo tengas en cuenta

- **Sesión 4.** Con `FUENTE = 'csv'` lee `BTCUSDT_12h_ohlc.csv`, que solamente trae BTCUSDT en 12h. Para otro mercado u otro intervalo toca usar `FUENTE = 'binance'`, que baja las velas de 1 minuto (unos 100 MB la primera vez). El tramo de prueba queda sellado: no se abre en este notebook.
- **Sesión 5.** Con `FUENTE = 'sintetica'` usa una cadena de opciones inventada. Con `FUENTE = 'deribit'` baja la cadena real de la API pública de Deribit y guarda una fila en `castle_labs/datos/opciones_diario.csv`. Los nombres de los campos de Deribit no se verificaron contra una respuesta real: si la descarga falla, el mensaje dice qué campos llegaron.
- **Las secciones 7 y 8 del notebook de opciones usan datos SIMULADOS.** Muestran cómo se hace la prueba. No dicen nada sobre el mercado.
- **En Colab los archivos se borran al cerrar la sesión.** Si llevas el registro diario ahí, descarga `opciones_diario.csv` cada día, o corre el notebook en tu computador.
- No hay respaldo académico de que un put wall o un call wall sea un soporte o una resistencia que se pueda operar. Lo que sí tiene respaldo es la relación entre el signo de la gamma agregada y el régimen de volatilidad. Por eso el muro se anota y se prueba antes de usarlo.
