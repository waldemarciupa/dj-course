package availability

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"tms-data-generator/generator/drivers"
	"tms-data-generator/generator/vehicles"
)

// GenerateDriverAvailability generates driver availability windows for a list of drivers.
func GenerateDriverAvailability(driversList []drivers.Driver, windowsPerDriver int) []DriverAvailability {
	availabilities := make([]DriverAvailability, 0, len(driversList)*windowsPerDriver)
	id := 1

	now := time.Now()
	startDate := now.AddDate(0, 0, -7) // Start from 7 days ago

	for _, driver := range driversList {
		for w := 0; w < windowsPerDriver; w++ {
			// Random date in the next 30 days
			daysOffset := rand.Intn(30)
			windowStart := startDate.AddDate(0, 0, daysOffset)

			// Random start hour (0-22)
			startHour := rand.Intn(23)
			windowStart = time.Date(windowStart.Year(), windowStart.Month(), windowStart.Day(), startHour, 0, 0, 0, windowStart.Location())

			// Duration: 2 to 10 hours
			durationHours := 2 + rand.Intn(9)
			windowEnd := windowStart.Add(time.Duration(durationHours) * time.Hour)

			// Status distribution: 70% available, 20% unavailable, 10% on_leave
			statusRand := rand.Intn(100)
			var status DriverAvailabilityStatus
			var notes string

			if statusRand < 70 {
				status = DriverAvailable
				notes = fmt.Sprintf("Available for delivery, %d hour shift", durationHours)
			} else if statusRand < 90 {
				status = DriverUnavailable
				notes = "Driver unavailable - training or meetings"
			} else {
				status = DriverOnLeave
				notes = "Scheduled leave"
			}

			availabilities = append(availabilities, DriverAvailability{
				ID:        id,
				DriverID:  driver.ID,
				StartTime: windowStart,
				EndTime:   windowEnd,
				Status:    status,
				Notes:     notes,
				CreatedAt: now,
			})
			id++
		}
	}

	return availabilities
}

// GenerateVehicleAvailability generates vehicle availability windows for a list of vehicles.
func GenerateVehicleAvailability(vehiclesList []vehicles.Vehicle, windowsPerVehicle int) []VehicleAvailability {
	availabilities := make([]VehicleAvailability, 0, len(vehiclesList)*windowsPerVehicle)
	id := 1

	now := time.Now()
	startDate := now.AddDate(0, 0, -7) // Start from 7 days ago

	for _, vehicle := range vehiclesList {
		for w := 0; w < windowsPerVehicle; w++ {
			// Random date in the next 30 days
			daysOffset := rand.Intn(30)
			windowStart := startDate.AddDate(0, 0, daysOffset)

			// Random start hour (0-22)
			startHour := rand.Intn(23)
			windowStart = time.Date(windowStart.Year(), windowStart.Month(), windowStart.Day(), startHour, 0, 0, 0, windowStart.Location())

			// Duration: 3 to 12 hours
			durationHours := 3 + rand.Intn(10)
			windowEnd := windowStart.Add(time.Duration(durationHours) * time.Hour)

			// Status distribution: 65% available, 20% reserved, 10% in_maintenance, 5% assigned
			statusRand := rand.Intn(100)
			var status VehicleAvailabilityStatus
			var location string
			var notes string

			if statusRand < 65 {
				status = VehicleAvailable
				location = randomLocation()
				notes = fmt.Sprintf("Available for deliveries at %s, %d hour slot", location, durationHours)
			} else if statusRand < 85 {
				status = VehicleReserved
				location = randomLocation()
				notes = "Vehicle reserved for specific orders"
			} else if statusRand < 95 {
				status = VehicleInMaintenance
				location = "Maintenance Center"
				notes = "Scheduled maintenance or inspection"
			} else {
				status = VehicleAssigned
				location = randomLocation()
				notes = "Vehicle assigned to active delivery route"
			}

			availabilities = append(availabilities, VehicleAvailability{
				ID:        id,
				VehicleID: vehicle.ID,
				StartTime: windowStart,
				EndTime:   windowEnd,
				Status:    status,
				Location:  location,
				Notes:     notes,
				CreatedAt: now,
			})
			id++
		}
	}

	return availabilities
}

// GenerateInsertStatementsDriver generates a single INSERT statement for driver availabilities.
func GenerateInsertStatementsDriver(availabilities []DriverAvailability) string {
	if len(availabilities) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(availabilities) * 250)
	sb.WriteString("INSERT INTO driver_availability (id, driver_id, start_time, end_time, status, notes, created_at) VALUES\n")

	for i, da := range availabilities {
		notes := escapeSQL(da.Notes)
		sb.WriteString("    (")
		sb.WriteString(strconv.Itoa(da.ID))
		sb.WriteString(", ")
		sb.WriteString(strconv.Itoa(da.DriverID))
		sb.WriteString(", '")
		sb.WriteString(da.StartTime.Format("2006-01-02 15:04:05"))
		sb.WriteString("', '")
		sb.WriteString(da.EndTime.Format("2006-01-02 15:04:05"))
		sb.WriteString("', '")
		sb.WriteString(string(da.Status))
		sb.WriteString("', '")
		sb.WriteString(notes)
		sb.WriteString("', '")
		sb.WriteString(da.CreatedAt.Format("2006-01-02 15:04:05"))
		sb.WriteString("')")

		if i < len(availabilities)-1 {
			sb.WriteString(",\n")
		} else {
			sb.WriteString(";\n")
		}
	}

	return sb.String()
}

// GenerateInsertStatementsVehicle generates a single INSERT statement for vehicle availabilities.
func GenerateInsertStatementsVehicle(availabilities []VehicleAvailability) string {
	if len(availabilities) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(availabilities) * 300)
	sb.WriteString("INSERT INTO vehicle_availability (id, vehicle_id, start_time, end_time, status, location, notes, created_at) VALUES\n")

	for i, va := range availabilities {
		location := escapeSQL(va.Location)
		notes := escapeSQL(va.Notes)
		sb.WriteString("    (")
		sb.WriteString(strconv.Itoa(va.ID))
		sb.WriteString(", ")
		sb.WriteString(strconv.Itoa(va.VehicleID))
		sb.WriteString(", '")
		sb.WriteString(va.StartTime.Format("2006-01-02 15:04:05"))
		sb.WriteString("', '")
		sb.WriteString(va.EndTime.Format("2006-01-02 15:04:05"))
		sb.WriteString("', '")
		sb.WriteString(string(va.Status))
		sb.WriteString("', '")
		sb.WriteString(location)
		sb.WriteString("', '")
		sb.WriteString(notes)
		sb.WriteString("', '")
		sb.WriteString(va.CreatedAt.Format("2006-01-02 15:04:05"))
		sb.WriteString("')")

		if i < len(availabilities)-1 {
			sb.WriteString(",\n")
		} else {
			sb.WriteString(";\n")
		}
	}

	return sb.String()
}

// Helper functions

func escapeSQL(s string) string {
	return strings.ReplaceAll(s, "'", "''")
}

func randomLocation() string {
	locations := []string{
		"Warehouse A - Downtown",
		"Warehouse B - Suburbs",
		"Warehouse C - Airport",
		"Distribution Center 1",
		"Distribution Center 2",
		"Service Station North",
		"Service Station South",
		"Hub Center Main",
		"Hub Center East",
		"Hub Center West",
	}
	return locations[rand.Intn(len(locations))]
}
