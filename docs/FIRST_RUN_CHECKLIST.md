# Checklist: pierwsze uruchomienie na Raspberry Pi

- Raspberry Pi OS Lite uruchamia sie headless i odpowiada po SSH
- `arecord -l` widzi karte USB audio
- mikrofon jest fizycznie podpiety do wejscia `mic-in`
- `configs/config.yaml` ma poprawne `audio.arecord_device`
- `.venv` istnieje i `pip install -r requirements.txt` zakonczyl sie sukcesem
- `python -m app.main check-audio` zwraca przechwycone bajty
- `python -m app.main init-db` utworzyl `data/db/audio_monitor.sqlite3`
- `scripts/smoke_test.sh` przechodzi
- `sudo systemctl status audio-monitor` pokazuje `active (running)`
- `journalctl -u audio-monitor -n 100` nie pokazuje petli bledow audio
- panel WWW odpowiada z innego urzadzenia w sieci domowej
- po restarcie Raspberry Pi usluga wraca automatycznie

