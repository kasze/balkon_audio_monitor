# Development Workflow

## Lokalna praca na WAV

```bash
./scripts/setup_venv.sh .venv
./scripts/install_python_deps.sh .venv
.venv/bin/python -m pytest
.venv/bin/python -m app.main --config configs/config.yaml analyze-dir sample_audio
.venv/bin/python -m app.main --config configs/config.yaml web
```

## Typowy cykl iteracji

1. Doloz albo podmien sample WAV w `sample_audio/`.
2. Uruchom `analyze-wav` albo `analyze-dir`.
3. Sprawdz dashboard i szczegoly eventow.
4. Dostrajal progi lub heurystyki.
5. Uruchom `pytest`.
6. Wypchnij zmiany na Raspberry Pi przez `scripts/deploy.sh`.

## Deploy na urzadzenie

```bash
./scripts/deploy.sh pi@raspberrypi.local /opt/audio-monitor
```

## Restart uslugi

```bash
./scripts/restart_service.sh pi@raspberrypi.local
```

## Podglad logow

```bash
./scripts/logs.sh pi@raspberrypi.local
```

## Rollback

Najprostszy rollback dla MVP:

1. Na laptopie przelacz repo na poprzedni stabilny commit.
2. Uruchom ponownie `./scripts/deploy.sh pi@raspberrypi.local /opt/audio-monitor`.
3. Zweryfikuj `journalctl` i `GET /health`.

Jesli chcesz rollback bez laptopa:

```bash
ssh pi@raspberrypi.local
cd /opt/audio-monitor
git log --oneline
git checkout <commit>
./scripts/install_python_deps.sh .venv
sudo systemctl restart audio-monitor
```

## Debugging

- `check-audio` diagnozuje ALSA i brak wejscia
- `run-live` pozwala izolowac pipeline bez weba
- `web` pozwala sprawdzic panel bez live capture
- `smoke_test.sh` daje szybki test integracyjny systemu
- baza SQLite jest w `data/db/audio_monitor.sqlite3`

## Co sprawdzac po zmianach

- czy pipeline nadal zapisuje `noise_intervals`
- czy jedna syrena nie rozbija sie na wiele eventow
- czy clipy zapisuja sie do `data/clips/`
- czy `GET /health` odpowiada
- czy restart systemd przywraca usluge po bledzie

