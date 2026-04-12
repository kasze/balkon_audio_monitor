# Raspberry Pi First Setup

Docelowy system: Raspberry Pi OS Lite, headless.

## 1. Przygotowanie systemu

1. Nagraj Raspberry Pi OS Lite na karte SD.
2. Wlacz SSH przed pierwszym startem.
3. Skonfiguruj siec i hostname.
4. Po zalogowaniu wykonaj:

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo raspi-config
```

W `raspi-config` ustaw:
- hostname
- timezone
- locale
- ewentualnie Wi-Fi

## 2. USB audio

Podlacz zewnetrzna karte dzwiekowa USB 5HV2 i mikrofon BOYA BY-MM1 z deadcatem.

Sprawdz ALSA:

```bash
arecord -l
arecord -L
```

Jesli chcesz miec stala nazwe urzadzenia, dostosuj [configs/asound.conf.example](/Users/kasze/audio_monitor/configs/asound.conf.example) i skopiuj go do:

```bash
cp configs/asound.conf.example ~/.asoundrc
```

Potem przetestuj:

```bash
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/test.wav
```

## 3. Katalog aplikacji

Zakladany katalog:

```bash
sudo mkdir -p /opt/audio-monitor
sudo chown pi:pi /opt/audio-monitor
```

## 4. Deploy

Z laptopa:

```bash
./scripts/deploy.sh pi@raspberrypi.local /opt/audio-monitor
```

## 5. Konfiguracja runtime

Na Raspberry Pi:

```bash
cd /opt/audio-monitor
cp configs/config.yaml.example configs/config.yaml
nano configs/config.yaml
```

Ustaw przynajmniej:
- `audio.arecord_device`, tylko jesli chcesz recznie wymusic urzadzenie; puste pole wlacza auto-detekcje USB capture
- `web.port`, jesli 8080 nie pasuje
- sciezki storage, jesli chcesz inne niz domyslne

## 6. Testy po pierwszym deployu

```bash
cd /opt/audio-monitor
.venv/bin/python -m app.main --config configs/config.yaml detect-audio
.venv/bin/python -m app.main --config configs/config.yaml check-audio
SKIP_AUDIO=0 ./scripts/smoke_test.sh configs/config.yaml
curl http://127.0.0.1:8080/health
```
