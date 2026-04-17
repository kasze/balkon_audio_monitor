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

Testy i codzienny deploy roboczy:

```bash
./scripts/test_and_deploy.sh
```

Jeśli potrzebujesz odświeżyć zależności systemowe, Python lub BirdNET:

```bash
./scripts/bootstrap_remote.sh
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

BAM łączy się z lokalnym serwerem BirdNET po adresie:

```yaml
classifier:
  birdnet_api_url: http://127.0.0.1:8081
  birdnet_timeout_seconds: 20.0
  birdnet_min_confidence: 0.20
  birdnet_num_results: 5
  birdnet_locale: pl
```

Port `8080` zostaje dla panelu BAM, dlatego BirdNET działa na `8081`.

Instalacja lokalnego serwera BirdNET jest ciężka, bo pobiera BirdNET-Analyzer, TensorFlow i zależności naukowe do osobnego `.birdnet-venv`. Na Raspberry Pi potrzebne jest kilka GB wolnego miejsca. Instalator trzyma też pliki tymczasowe na dysku w katalogu repo, a nie w `/tmp`, bo `/tmp` bywa mały i jest montowany jako `tmpfs`. Żeby włączyć instalację podczas deployu, ustaw lokalnie w `configs/deploy.env`:

```bash
AUDIO_MONITOR_INSTALL_BIRDNET=1
```

Potem uruchom:

```bash
./scripts/deploy.sh
```

Sprawdzenie usługi na Raspberry Pi:

```bash
sudo systemctl status birdnet-server
journalctl -u birdnet-server -n 100 --no-pager
```

Jeśli instalujesz unit ręcznie, pamiętaj żeby `User=` w `birdnet-server.service` wskazywał istniejącego użytkownika na Raspberry Pi, np. `kasze`, a nie domyślne `pi`.

## Pliki i katalogi

- `configs/config.yaml` lokalna konfiguracja aplikacji
- `configs/deploy.env` lokalna konfiguracja deployu, ignorowana przez Git
- `data/db/` baza SQLite
- `data/clips/` zapisane próbki audio i obrazy widma
- `systemd/audio-monitor.service` jednostka usługi
- `systemd/birdnet-server.service` jednostka lokalnego API BirdNET
