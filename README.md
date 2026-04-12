# Balkon Audio Monitor (BAM)

Aplikacja do ciągłego monitoringu audio na Raspberry Pi z panelem WWW, zapisem zdarzeń i klasyfikacją dźwięków.

## Wymagania

- Raspberry Pi z Raspberry Pi OS
- karta microSD
- zasilanie i sieć Wi‑Fi lub Ethernet
- mikrofon / karta audio USB
- komputer do przygotowania karty SD i deployu przez SSH

## Przygotowanie karty SD

1. Zainstaluj `Raspberry Pi Imager`.
2. Wybierz system: `Raspberry Pi OS Lite`.
3. Wybierz kartę SD.
4. Wejdź w `Edit Settings` przed nagraniem obrazu.
5. Ustaw:
   - hostname, np. `raspberrypi`
   - włączenie `SSH`
   - nazwę użytkownika i hasło do SSH
   - dane Wi‑Fi: SSID, hasło, kraj
   - strefę czasową i układ klawiatury, jeśli chcesz
6. Nagraj obraz na kartę.
7. Włóż kartę do Raspberry Pi i uruchom urządzenie.

Po starcie połącz się:

```bash
ssh pi@raspberrypi.local
```

albo po IP:

```bash
ssh pi@192.168.x.x
```

## Przygotowanie repo lokalnie

W katalogu projektu:

```bash
./scripts/setup_venv.sh .venv
./scripts/install_python_deps.sh .venv
cp configs/config.yaml.example configs/config.yaml
.venv/bin/python -m app.main --config configs/config.yaml init-db
```

Testy lokalne:

```bash
./scripts/test.sh
```

## Konfiguracja aplikacji

Główny plik konfiguracyjny:

```bash
configs/config.yaml
```

Najważniejsze pola:

- `audio.arecord_device`
  - zostaw puste, jeśli aplikacja ma sama wybrać urządzenie
  - ustaw np. `plughw:2,0`, jeśli chcesz wymusić konkretne wejście
- `storage.database_path`
- `storage.clip_dir`
- `web.port`
- `classifier.birdnet_api_url`
  - opcjonalne
  - jeśli ustawione, BAM wyśle próbkę do BirdNET dla klas ptasich wykrytych przez YAMNet

## Sprawdzenie audio

Na Raspberry Pi:

```bash
arecord -l
```

Z repo:

```bash
.venv/bin/python -m app.main --config configs/config.yaml detect-audio
.venv/bin/python -m app.main --config configs/config.yaml check-audio
```

## Uruchamianie lokalne

Analiza pojedynczego WAV:

```bash
.venv/bin/python -m app.main --config configs/config.yaml analyze-wav sample_audio/demo_siren.wav
```

Analiza katalogu WAV:

```bash
.venv/bin/python -m app.main --config configs/config.yaml analyze-dir sample_audio
```

Panel WWW:

```bash
.venv/bin/python -m app.main --config configs/config.yaml web
```

Pełna usługa lokalnie:

```bash
.venv/bin/python -m app.main --config configs/config.yaml service
```

## Deploy na Raspberry Pi

Skonfiguruj lokalny plik deployu:

```bash
cp configs/deploy.env.example configs/deploy.env
```

Ustaw co najmniej:

```bash
AUDIO_MONITOR_TARGET=pi@raspberrypi.local
AUDIO_MONITOR_REMOTE_DIR=/opt/audio-monitor
```

Jeśli używasz hasła SSH:

```bash
AUDIO_MONITOR_SSH_PASSWORD='twoje-haslo'
```

Deploy:

```bash
./scripts/deploy.sh
```

Testy i deploy jednym poleceniem:

```bash
./scripts/test_and_deploy.sh
```

## Uruchomienie jako usługa

Deploy instaluje jednostkę `systemd` i restartuje usługę automatycznie.

Ręczna obsługa na Raspberry Pi:

```bash
sudo systemctl status audio-monitor
sudo systemctl restart audio-monitor
sudo systemctl enable audio-monitor
```

## Logi

Z repo:

```bash
./scripts/logs.sh
```

Zdalnie:

```bash
./scripts/logs.sh pi@raspberrypi.local
```

Na Raspberry Pi:

```bash
journalctl -u audio-monitor -f
```

## Smoke test

```bash
./scripts/smoke_test.sh configs/config.yaml
```

Bez testu wejścia audio:

```bash
SKIP_AUDIO=1 ./scripts/smoke_test.sh configs/config.yaml
```

## Panel WWW

Domyślnie:

```text
http://<adres-rpi>:8080
```

Endpoint zdrowia:

```text
/health
```

## BirdNET

Opcjonalna integracja działa tylko dla przypadków, gdy YAMNet wykryje klasy ptasie. Wtedy BAM może wysłać próbkę do serwera BirdNET API i zapisać gatunek.

Konfiguracja:

```yaml
classifier:
  birdnet_api_url: http://host:port
  birdnet_timeout_seconds: 15.0
  birdnet_min_confidence: 0.20
  birdnet_num_results: 5
  birdnet_locale: pl
```

## Pliki i katalogi

- `configs/config.yaml` lokalna konfiguracja aplikacji
- `configs/deploy.env` lokalna konfiguracja deployu, ignorowana przez Git
- `data/db/` baza SQLite
- `data/clips/` zapisane próbki audio i obrazy widma
- `systemd/audio-monitor.service` jednostka usługi
