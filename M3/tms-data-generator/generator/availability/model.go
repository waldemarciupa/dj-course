package availability

import "time"

// DriverAvailabilityStatus represents the availability status of a driver.
type DriverAvailabilityStatus string

const (
	DriverAvailable   DriverAvailabilityStatus = "available"
	DriverUnavailable DriverAvailabilityStatus = "unavailable"
	DriverOnLeave     DriverAvailabilityStatus = "on_leave"
	DriverAssigned    DriverAvailabilityStatus = "assigned"
)

// VehicleAvailabilityStatus represents the availability status of a vehicle.
type VehicleAvailabilityStatus string

const (
	VehicleAvailable     VehicleAvailabilityStatus = "available"
	VehicleInMaintenance VehicleAvailabilityStatus = "in_maintenance"
	VehicleReserved      VehicleAvailabilityStatus = "reserved"
	VehicleAssigned      VehicleAvailabilityStatus = "assigned"
)

// DriverAvailability represents a time window of driver availability.
type DriverAvailability struct {
	ID        int
	DriverID  int
	StartTime time.Time
	EndTime   time.Time
	Status    DriverAvailabilityStatus
	Notes     string
	CreatedAt time.Time
}

// VehicleAvailability represents a time window of vehicle availability.
type VehicleAvailability struct {
	ID        int
	VehicleID int
	StartTime time.Time
	EndTime   time.Time
	Status    VehicleAvailabilityStatus
	Location  string
	Notes     string
	CreatedAt time.Time
}
