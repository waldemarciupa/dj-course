# TMS Data Generator — Rozbudowa o Dostępność Kierowców i Pojazdów

## Przegląd zmian

Projekt `tms-data-generator` rozbudowany został o system śledowania dostępności (availability) kierowców i pojazdów. Nowe tabele umożliwiają modelowanie czasowych okien dostępności w systemie zarządzania transportem.

## Nowe tabele DDL

### 1. `driver_availability`
Przechowuje okna czasowe dostępności kierowców.

```sql
CREATE TABLE driver_availability (
    id INT PRIMARY KEY,
    driver_id INT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('available', 'unavailable', 'on_leave', 'assigned')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE CASCADE
);

CREATE INDEX idx_driver_availability_driver_id ON driver_availability(driver_id);
CREATE INDEX idx_driver_availability_status ON driver_availability(status);
```

**Statusy:**
- `available` — kierowca dostępny do pracy
- `unavailable` — kierowca niedostępny (szkolenia, spotkania)
- `on_leave` — urlop zaplanowany
- `assigned` — przydzielony do konkretnego zlecenia

### 2. `vehicle_availability`
Przechowuje okna czasowe dostępności pojazdów.

```sql
CREATE TABLE vehicle_availability (
    id INT PRIMARY KEY,
    vehicle_id INT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('available', 'in_maintenance', 'reserved', 'assigned')),
    location VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE
);

CREATE INDEX idx_vehicle_availability_vehicle_id ON vehicle_availability(vehicle_id);
CREATE INDEX idx_vehicle_availability_status ON vehicle_availability(status);
```

**Statusy:**
- `available` — pojazd dostępny do przydzielenia
- `in_maintenance` — przegląd techniczny lub naprawa
- `reserved` — zarezerwowany dla konkretnych zleceń
- `assigned` — przydzielony do aktywnej trasy dostawy

## Powiązania z innymi tabelami

### Tabela `transportation_orders` — rozszerzenie
Dodano dwie opcjonalne kolumny umożliwiające przypisanie kierowcy i pojazdu:

```sql
ALTER TABLE transportation_orders ADD COLUMN assigned_driver_id INT REFERENCES drivers(id);
ALTER TABLE transportation_orders ADD COLUMN assigned_vehicle_id INT REFERENCES vehicles(id);
```

## Architektura generacji (DML)

### Nowy pakiet Go: `generator/availability`

Struktura plików:
```
generator/availability/
├── model.go         # Struktury DriverAvailability, VehicleAvailability + enumeracje statusów
└── availability.go  # Funkcje generacji + SQL statements
```

### Konfiguracja (`generator/config/count.go`)
```go
const (
    DRIVER_AVAIL_WINDOWS_PER_DRIVER = 7      // 7 okien dostępności na kierowcę
    VEHICLE_AVAIL_WINDOWS_PER_VEHICLE = 5    // 5 okien dostępności na pojazd
)
```

**Liczby rekordów:**
- Driver Availability: 20 kierowców × 7 = **140 rekordów**
- Vehicle Availability: 50 pojazdów × 5 = **250 rekordów**

### Parametry generacji
- **Okna czasowe**: losowe, począwszy od 7 dni wstecz do 30 dni do przodu
- **Czasy**: losowy początek (0-22) i czas trwania (2-10 godzin dla kierowców, 3-12 dla pojazdów)
- **Statusy** (rozkład):
  - Kierowcy: 70% available, 20% unavailable, 10% on_leave
  - Pojazdy: 65% available, 20% reserved, 10% in_maintenance, 5% assigned
- **Lokalizacje**: losowy wybór z listy 10 magazynów/centrów dystrybucji
- **SQL escaping**: `strings.ReplaceAll(s, "'", "''")` dla bezpieczeństwa

## Fazy generacji (aktualizacja `generator/generator.go`)

1. **Phase 1** — Generacja niezależna (parallel): vehicles, drivers, customers
2. **Phase 2** — Generacja dostępności (sekwencyjna, zależy od phase 1)
   - `GenerateDriverAvailability(driversList, windowsPerDriver)`
   - `GenerateVehicleAvailability(vehiclesList, windowsPerVehicle)`
3. **Phase 3-7** — Pozostałe: orders, order items, timeline events, SQL statements

## Format wyjścia SQL

```sql
INSERT INTO driver_availability (id, driver_id, start_time, end_time, status, notes, created_at) VALUES
    (1, 1, '2025-12-24 15:00:00', '2025-12-24 23:00:00', 'on_leave', 'Scheduled leave', '2025-12-21 13:05:15'),
    (2, 1, '2025-12-28 21:00:00', '2025-12-29 04:00:00', 'available', 'Available for delivery, 7 hour shift', '2025-12-21 13:05:15'),
    ...;
```

## Uruchomienie

```bash
cd M3/tms-data-generator
go run cmd/tms-data-generator/main.go
# Wygenerowany plik: output/tms-latest.sql (~13,800 linii)
```

## Użyteczne zapytania testowe (przykłady)

```sql
-- Dostępni kierowcy w określonym przedziale czasowym
SELECT d.*, da.* FROM drivers d
JOIN driver_availability da ON d.id = da.driver_id
WHERE da.status = 'available' 
  AND da.start_time <= NOW() 
  AND da.end_time >= NOW();

-- Pojazdy dostępne do przydział w magazynie
SELECT v.*, va.* FROM vehicles v
JOIN vehicle_availability va ON v.id = va.vehicle_id
WHERE va.status IN ('available', 'reserved')
  AND va.location LIKE '%Downtown%';

-- Zlecenia z przydzielonymi kierowcami i pojazdam
SELECT * FROM transportation_orders
WHERE assigned_driver_id IS NOT NULL 
  AND assigned_vehicle_id IS NOT NULL;
```

## Rozszerzenia przyszłe

1. **Optymalizacja przydzielania**: Algorytm automatycznego przydzielania kierowcy/pojazdu do zlecenia na podstawie dostępności
2. **Konflikty**: Walidacja czy przedziały czasu się nie nakładają
3. **Historia zmian**: Tabela `availability_audit` śledząca zmiany statusów
4. **Dane w czasie rzeczywistym**: Integracja z systemami GPS/IoT
