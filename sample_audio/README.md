# sample_audio

Ten katalog sluzy do lokalnych testow offline bez mikrofonu.

Zalecany format probek:
- WAV PCM 16-bit
- mono
- 16 kHz
- nazwy plikow opisowe, np. `ambulance_2026-04-12_0900.wav`

Szybka konwersja z dowolnego WAV/MP3:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 sample_audio/output.wav
```

Repo zawiera dwa syntetyczne sample demonstracyjne wygenerowane skryptem `sample_audio/generate_samples.py`.

