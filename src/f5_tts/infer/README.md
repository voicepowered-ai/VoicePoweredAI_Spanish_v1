# VoicePowered v1

Presentamos VoicePowered v1, un modelo Text to Speech open source centrado en la clonación de voces con acento español. Este modelo es un fine-tuning del modelo F5-TTS (https://huggingface.co/SWivid/F5-TTS)

Ofrecemos una app Gradio donde probar el modelo. Esta app incluye generación de una voz y generación multi-voz.

Para la selección de los audios de referencia, por favor sigan las siguientes pautas.

- El audio de referencia debe durar menos de 15 segundos. Dentro de estos 15 segundos, debe haber un pequeño silencio al principio y al final del audio. Preferiblemente, dentro del audio debe haber frases completas.
- El modelo utilizará las características de este audio para la generación, incluyendo la velocidad del habla, la prosodia, la emoción, las pronunciaciones, e incluso la calidad del audio, por lo que hay que asegurarse que se utiliza un audio de referencia con las características del audio deseado.

## Gradio App

Currently supported features:

- Clonación de voz simple
- Generación multi-voz