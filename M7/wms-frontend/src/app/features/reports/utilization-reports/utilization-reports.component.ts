
import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DropdownComponent } from '../../../ui-library/Dropdown.component';
import { Heading3Component, Heading4Component } from '../../../ui-library/Typography/Typography.component';
import { UtilizationReport } from '../reports.model';

@Component({
  selector: 'app-utilization-reports',
  standalone: true,
  imports: [CommonModule, FormsModule, DropdownComponent, Heading3Component, Heading4Component],
  template: `
    <div class="p-6">
      <div class="flex justify-between items-center mb-6">
        <ui-heading3>Utilization Reports</ui-heading3>
        <div class="flex space-x-3">
          <ui-dropdown
            label="Period"
            [options]="[
              { value: 'week', label: 'This Week' },
              { value: 'month', label: 'This Month' },
              { value: 'quarter', label: 'This Quarter' },
              { value: 'year', label: 'This Year' }
            ]"
            [value]="period"
            (valueChange)="periodChange.emit($event)"
          />
          <button (click)="export.emit()" class="btn btn-secondary">
            <svg class="h-4 w-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Export
          </button>
        </div>
      </div>

      <!-- Utilization Categories -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Space Utilization -->
        <div class="card p-6">
          <ui-heading4>Space Utilization</ui-heading4>
          <div class="space-y-4">
            @for (zone of report?.spaceUtilization; track zone.zoneName) {
              <div class="space-y-2">
                <div class="flex justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">{{ zone.zoneName }}</span>
                  <span class="text-sm font-medium text-gray-900 dark:text-white">{{ zone.utilization }}%</span>
                </div>
                <div class="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
                  <div [class]="getUtilizationBarClass(zone.utilization)"
                        class="h-2 rounded-full transition-all"
                        [style.width.%]="zone.utilization"></div>
                </div>
              </div>
            }
          </div>
        </div>

        <!-- Equipment Utilization -->
        <div class="card p-6">
          <ui-heading4>Equipment Utilization</ui-heading4>
          <div class="space-y-4">
            @for (equipment of report?.equipmentUtilization; track equipment.equipmentType) {
              <div class="space-y-2">
                <div class="flex justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">{{ equipment.equipmentType }}</span>
                  <span class="text-sm font-medium text-gray-900 dark:text-white">{{ equipment.utilization }}%</span>
                </div>
                <div class="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
                  <div [class]="getUtilizationBarClass(equipment.utilization)"
                        class="h-2 rounded-full transition-all"
                        [style.width.%]="equipment.utilization"></div>
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400">
                  {{ equipment.activeHours }}h active / {{ equipment.totalHours }}h total
                </div>
              </div>
            }
          </div>
        </div>

        <!-- Personnel Utilization -->
        <div class="card p-6">
          <ui-heading4>Personnel Utilization</ui-heading4>
          <div class="space-y-4">
            @for (personnel of report?.personnelUtilization; track personnel.role) {
              <div class="space-y-2">
                <div class="flex justify-between">
                  <span class="text-sm text-gray-600 dark:text-gray-400">{{ personnel.role }}</span>
                  <span class="text-sm font-medium text-gray-900 dark:text-white">{{ personnel.utilization }}%</span>
                </div>
                <div class="w-full bg-gray-200 dark:bg-dark-700 rounded-full h-2">
                  <div [class]="getUtilizationBarClass(personnel.utilization)"
                        class="h-2 rounded-full transition-all"
                        [style.width.%]="personnel.utilization"></div>
                </div>
                <div class="text-xs text-gray-500 dark:text-gray-400">
                  {{ personnel.activeEmployees }} / {{ personnel.totalEmployees }} employees
                </div>
              </div>
            }
          </div>
        </div>
      </div>

      <!-- Detailed Utilization Table -->
      <div class="card p-6 mt-6">
        <ui-heading4>Detailed Utilization Breakdown</ui-heading4>
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200 dark:divide-dark-700">
            <thead class="bg-gray-50 dark:bg-dark-800">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Resource</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Type</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Capacity</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Used</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Utilization</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody class="bg-white dark:bg-dark-800 divide-y divide-gray-200 dark:divide-dark-700">
              @for (item of report?.detailedBreakdown; track item.name) {
                <tr class="hover:bg-gray-50 dark:hover:bg-dark-700">
                  <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">{{ item.name }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{{ item.type }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{{ item.capacity }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{{ item.used }}</td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center">
                      <div class="w-16 bg-gray-200 dark:bg-dark-700 rounded-full h-2 mr-2">
                        <div [class]="getUtilizationBarClass(item.utilization)"
                              class="h-2 rounded-full"
                              [style.width.%]="item.utilization"></div>
                      </div>
                      <span class="text-sm text-gray-900 dark:text-white">{{ item.utilization }}%</span>
                    </div>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <span [class]="getUtilizationStatusClass(item.utilization)"
                          class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium">
                      {{ getUtilizationStatus(item.utilization) }}
                    </span>
                  </td>
                </tr>
              }
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `,
})
export class UtilizationReportsComponent {
  @Input() report: UtilizationReport | null = null;
  @Input() period = 'month';
  @Output() periodChange = new EventEmitter<string>();
  @Output() export = new EventEmitter<void>();

  getUtilizationBarClass(utilization: number): string {
    if (utilization >= 90) return 'bg-error-500';
    if (utilization >= 75) return 'bg-warning-500';
    if (utilization >= 50) return 'bg-primary-500';
    return 'bg-success-500';
  }

  getUtilizationStatusClass(utilization: number): string {
    if (utilization >= 90) return 'bg-error-100 text-error-800';
    if (utilization >= 75) return 'bg-warning-100 text-warning-800';
    if (utilization >= 50) return 'bg-primary-100 text-primary-800';
    return 'bg-success-100 text-success-800';
  }

  getUtilizationStatus(utilization: number): string {
    if (utilization >= 90) return 'Critical';
    if (utilization >= 75) return 'High';
    if (utilization >= 50) return 'Moderate';
    return 'Low';
  }
}
