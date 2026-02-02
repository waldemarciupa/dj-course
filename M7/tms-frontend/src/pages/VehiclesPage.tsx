
import React from 'react';
import { useVehiclesList } from '../hooks/queries';
import { LoadingPage, ErrorMessage } from '../components';
import LiveFleetMap from './vehicles/LiveFleetMap';
import PageHeader from './vehicles/PageHeader';
import VehicleFleetDisplay from './vehicles/VehicleFleetDisplay';


const VehiclesPage = () => {
  const { data: vehicles = [], isLoading, error, refetch } = useVehiclesList();

  const handleRetry = () => {
    refetch();
  };

  if (isLoading) {
    return <LoadingPage />;
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <ErrorMessage
          error={error instanceof Error ? error.message : 'Failed to load vehicles'}
          onRetry={handleRetry}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader />
      <LiveFleetMap vehicles={vehicles} />
      <VehicleFleetDisplay vehicles={vehicles} />
    </div>
  );
};

export default VehiclesPage;
