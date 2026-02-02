import React, { useState } from 'react';
import { Vehicle } from '../../model/vehicles';
import { VehicleFilters } from './VehicleFilters';
import { VehiclesList } from './VehiclesList';
import { VehiclesTable } from './VehiclesTable';

interface VehicleFleetDisplayProps {
  vehicles: Vehicle[];
}

const VehicleFleetDisplay: React.FC<VehicleFleetDisplayProps> = ({ vehicles }) => {
  const [view, setView] = useState<'grid' | 'table'>('grid');

  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | Vehicle['status']>('all');
  const [typeFilter, setTypeFilter] = useState<'all' | Vehicle['type']>('all');
  const [ownershipFilter, setOwnershipFilter] = useState<'all' | Vehicle['ownership']['type']>('all');

  const filteredVehicles = vehicles.filter(vehicle => {
    const matchesSearch =
      vehicle.plateNumber.toLowerCase().includes(searchTerm.toLowerCase()) ||
      vehicle.make.toLowerCase().includes(searchTerm.toLowerCase()) ||
      vehicle.model.toLowerCase().includes(searchTerm.toLowerCase()) ||
      vehicle.currentDriver?.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus = statusFilter === 'all' || vehicle.status === statusFilter;
    const matchesType = typeFilter === 'all' || vehicle.type === typeFilter;
    const matchesOwnership = ownershipFilter === 'all' || vehicle.ownership.type === ownershipFilter;

    return matchesSearch && matchesStatus && matchesType && matchesOwnership;
  });

  const handleClearFilters = () => {
    setSearchTerm('');
    setStatusFilter('all');
    setTypeFilter('all');
    setOwnershipFilter('all');
  };

  const hasActiveFilters = searchTerm !== '' || statusFilter !== 'all' || typeFilter !== 'all' || ownershipFilter !== 'all';

  return (
    <div className="space-y-4">
      <VehicleFilters
        searchTerm={searchTerm}
        statusFilter={statusFilter}
        typeFilter={typeFilter}
        ownershipFilter={ownershipFilter}
        view={view}
        onSearchChange={setSearchTerm}
        onStatusChange={setStatusFilter}
        onTypeChange={setTypeFilter}
        onOwnershipChange={setOwnershipFilter}
        onViewChange={setView}
        onClearFilters={handleClearFilters}
        hasActiveFilters={hasActiveFilters}
        resultCount={filteredVehicles.length}
      />

      {view === 'grid' ? (
        <VehiclesList vehicles={filteredVehicles} />
      ) : (
        <VehiclesTable vehicles={filteredVehicles} />
      )}
    </div>
  );
};

export default VehicleFleetDisplay;
