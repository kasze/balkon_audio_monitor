# Audio Monitor MVP

Lekki projekt do 24/7 monitoringu dzwiekow na balkonie na Raspberry Pi 3 B+ z USB audio i panelem WWW. MVP celowo stawia na stabilnosc, prosty deploy przez SSH/rsync i czytelna architekture, a nie na maksymalna dokladnosc klasyfikacji.

## Co robi MVP

- przechwytuje audio live przez ALSA `arecord` albo analizuje pliki WAV offline
- liczy lekkie cechy audio na ramkach 0.5 s
- wykrywa epizody przez adaptacyjny prog energii z histereza
- scala sasiednie ramki w jedno zdarzenie
- klasyfikuje zdarzenia lokalnie przez YAMNet uruchamiany w LiteRT do kategorii:
  - `ambulance`
  - `police`
  - `fire_truck`
  - `airplane`
  - `helicopter`
  - `street_background`
- cache'uje podobne klipy w SQLite, zeby nie przepalac CPU na wielokrotnym rozpoznawaniu bardzo podobnych zdarzen i zeby przygotowac grunt pod przyszly reuse dla BirdNET / zewnetrznych API
- zapisuje interwaly halasu, zdarzenia, clipy i decyzje klasyfikatora do SQLite
- rotuje stare clipy WAV po wieku i rozmiarze oraz pilnuje minimalnego zapasu wolnego miejsca na dysku
- udostepnia panel WWW z dashboardem, szczegolami zdarzen i odsluchem clipow
- wystawia endpoint zdrowia `GET /health`

## Dlaczego taka architektura

- `arecord` zamiast PortAudio/PyAudio: natywnie pasuje do ALSA i zwykle zachowuje sie stabilniej na headless Raspberry Pi.
- `numpy` + jedna FFT na ramke: to jest realne dla RPi 3 B+ i wystarcza do MVP.
- YAMNet idzie lokalnie przez LiteRT, bez chmury i bez quota API. Cache podobienstwa sluzy do oszczedzania CPU teraz i do przyszlego reuse klasyfikacji dla kolejnych modeli/API.
- Flask + Waitress + server-side HTML: niski narzut RAM/CPU i zero SPA.
- SQLite: lokalnie, bez zaleznosci od chmury i bez zbednej infrastruktury.

## Struktura repo

- `app/` kod aplikacji
- `tests/` testy jednostkowe i integracyjne
- `scripts/` skrypty operacyjne i deploy
- `configs/` konfiguracja YAML i przyklad ALSA
- `systemd/` jednostka uslugi
- `docs/` dokumentacja wdrozenia i checklisty
- `sample_audio/` probki offline i generator demo

## Moduly aplikacji

- `app/capture` live capture przez `arecord` oraz pliki WAV
- `app/features` lekkie cechy audio
- `app/detect` detekcja epizodow z histereza
- `app/classify` YAMNet LiteRT, cache podobienstwa i fallback heurystyczny
- `app/aggregate` laczenie ramek w interwaly halasu i zdarzenia
- `app/storage` SQLite i clipy WAV
- `app/web` panel WWW
- `app/config.py` konfiguracja YAML
- `app/logging_setup.py` logi do stdout/journald

## Szybki start lokalnie

```bash
./scripts/setup_venv.sh .venv
./scripts/install_python_deps.sh .venv
.venv/bin/python -m app.main --config configs/config.yaml init-db
.venv/bin/python -m app.main --config configs/config.yaml analyze-dir sample_audio
.venv/bin/python -m app.main --config configs/config.yaml web
```

Panel: [http://127.0.0.1:8080](http://127.0.0.1:8080)

## Tryb offline

Pojedynczy plik WAV:

```bash
.venv/bin/python -m app.main --config configs/config.yaml analyze-wav sample_audio/demo_siren.wav
```

Katalog z WAV:

```bash
.venv/bin/python -m app.main --config configs/config.yaml analyze-dir sample_audio
```

Wymagany format offline dla MVP:
- WAV PCM 16-bit
- 16 kHz
- mono lub stereo

Jesli probka ma inny format, skonwertuj ja przed uruchomieniem:

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 sample_audio/output.wav
```

## Live capture na Raspberry Pi

1. Sprawdz karte audio:

```bash
arecord -l
```

2. Domyslnie zostaw `audio.arecord_device` puste. Aplikacja sama wybierze pierwsze sensowne wejscie capture, preferujac USB i uzywajac `plughw:X,Y`. Jesli chcesz wymusic konkretne urzadzenie, ustaw je recznie w [configs/config.yaml.example](/Users/kasze/audio_monitor/configs/config.yaml.example), np. `plughw:2,0`.

3. Probe input:

```bash
.venv/bin/python -m app.main --config configs/config.yaml detect-audio
.venv/bin/python -m app.main --config configs/config.yaml check-audio
```

Jesli usluga juz trzyma input, `check-audio` zwroci komunikat o zajetym urzadzeniu zamiast falszywego bledu konfiguracji.

## YAMNet

- backend klasyfikacji jest ustawiony domyslnie na `classifier.backend: yamnet`
- model `.tflite` i `yamnet_class_map.csv` pobieraja sie automatycznie przy pierwszej klasyfikacji do `models/`
- na Raspberry Pi 3 B+ inference jest ograniczony do fragmentu zdarzenia i limitu okien, zeby nie mielic calego 90-sekundowego clipu
- jesli YAMNet albo LiteRT nie sa dostepne, aplikacja fallbackuje do heurystyki i zapisuje powod w `classifier_decisions.details`
- cache podobienstwa audio siedzi w SQLite i pozwala reuse'owac decyzje dla niemal identycznych klipow

4. Uruchom bez weba:

```bash
.venv/bin/python -m app.main --config configs/config.yaml run-live
```

5. Uruchom wszystko jako jedna usluge:

```bash
.venv/bin/python -m app.main --config configs/config.yaml service
```

## Deploy na Raspberry Pi

Przykladowy deploy do `/opt/audio-monitor`:

```bash
./scripts/deploy.sh pi@raspberrypi.local /opt/audio-monitor
```

Skrypt:
- synchronizuje repo przez `rsync`
- instaluje zaleznosci systemowe
- tworzy `.venv`
- instaluje zaleznosci Pythona
- kopiuje jednostke systemd
- wlacza i restartuje usluge

## Systemd i logi

Jednostka: [systemd/audio-monitor.service](/Users/kasze/audio_monitor/systemd/audio-monitor.service)

Logi lokalnie:

```bash
./scripts/logs.sh
```

Logi z laptopa:

```bash
./scripts/logs.sh pi@raspberrypi.local
```

Auto-detekcja inputu:

```bash
.venv/bin/python -m app.main --config configs/config.yaml detect-audio
```

Restart:

```bash
./scripts/restart_service.sh pi@raspberrypi.local
```

## Smoke test

Skrypt sprawdza baze, probe audio oraz panel WWW:

```bash
./scripts/smoke_test.sh configs/config.yaml
```

Na laptopie bez mikrofonu USB:

```bash
SKIP_AUDIO=1 ./scripts/smoke_test.sh configs/config.yaml
```

## Retencja clipow

- `storage.clip_max_megabytes` ogranicza laczny rozmiar `data/clips/`
- `storage.clip_max_age_days` kasuje stare clipy po wieku
- `storage.min_free_disk_megabytes` trzyma rezerwe wolnego miejsca na filesystemie
- przy retencji usuwane sa tylko pliki WAV i ich metadane; eventy, statystyki i decyzje klasyfikatora zostaja w SQLite
- jesli limitu nie da sie spelnic nawet po cleanupie, nowy clip nie zostanie zapisany, ale event nadal trafi do bazy

## Zbieranie probek do dalszego ulepszania

- nagrywaj osobne sample przez [scripts/record_sample.sh](/Users/kasze/audio_monitor/scripts/record_sample.sh)
- wrzucaj je do `sample_audio/`
- notuj rzeczywista etykiete w nazwie pliku albo w osobnym arkuszu
- uruchamiaj `analyze-dir`, porownuj wyniki i iteracyjnie strojenie heurystyk

## Ograniczenia MVP

- mapowanie klas YAMNet do naszych kategorii jest warstwa translacji nad AudioSet, wiec nadal wymaga strojenia na realnych probkach z balkonu
- brak resamplingu w aplikacji: dla stabilnosci MVP oczekuje WAV 16 kHz
- dashboard pokazuje srednie godzinowe zamiast gestego wykresu z sekundowych punktow
- jedna usluga laczy ingest i web, zeby uproscic operacje; gdy projekt dojrzeje, mozna je rozdzielic

## Kolejny krok po MVP

Najczystsza sciezka rozwoju to dodanie drugiego klasyfikatora za interfejsem z `app/classify`, np. lekkiego TFLite uruchamianego tylko na zakonczonych eventach zamiast na calym strumieniu.
