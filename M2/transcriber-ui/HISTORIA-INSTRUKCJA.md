# Historia Transkrypcji - Instrukcja użytkownika

## Nowa funkcjonalność: Zakładka "Transcription History"

### Co zostało dodane?

Aplikacja zawiera nową zakładkę do przeglądania historii wszystkich zapisanych transkrypcji.

### Jak to działa?

#### Sidebar (Lewa strona)

- **Lista transkrypcji** - wszystkie zapisane transkrypcje, posortowane od najnowszych
- **Tekst ucinany** - długie transkrypcje są skracane do 40 znaków z "..." na końcu
- **Kliknij element** - aby zobaczyć pełne szczegóły po prawej stronie

#### Szczegóły (Prawa strona)

Po wybraniu transkrypcji zobaczysz:

- Nazwę pliku audio
- Datę i godzinę (format: `29 November 3:03 pm`)
- Pełny tekst transkrypcji

#### Przycisk Play

- **Włączony** - gdy wybierzesz transkrypcję
- **Wyłączony** - gdy nic nie wybierzesz
- **Kliknięcie** - odtwarza plik audio .wav
- **Podczas odtwarzania** - przycisk pokazuje "⏸ Playing..."

### Wymagania

Upewnij się, że zainstalowałeś `pygame`:

```bash
pip install -r requirements.txt
```

Lub jeśli masz już zainstalowane inne pakiety:

```bash
pip install pygame
```

### Testowanie

Aby przetestować funkcjonalność bez nagrywania nowych transkrypcji:

```bash
python test_history.py
```

To stworzy kilka przykładowych transkrypcji w folderze `output/`.

### Techniczne notatki

- Transkrypcje są przechowywane jako pary plików: `.wav` (audio) i `.json` (metadata)
- Historia jest ładowana dynamicznie z folderu `output/`
- Aplikacja nie wymaga restartu aby zobaczyć nowe transkrypcje
- Formatowanie daty pracuje na wszystkich systemach (Windows, macOS, Linux)

### Struktura folderu output

```
output/
├── recording-1764411328.wav      # Plik audio
├── recording-1764411328.json     # Metadane
├── recording-1704067800.wav
├── recording-1704067800.json
└── ... (więcej plików)
```

### Format JSON metadanych

```json
{
  "audio_file": "recording-1764411328.wav",
  "transcription": "Transkrypcja tekstu...",
  "timestamp": "2025-11-29 11:15:31"
}
```

---

Pytania lub problemy? Sprawdź plik `transcriber.log` aby zobaczyć detale błędów.
