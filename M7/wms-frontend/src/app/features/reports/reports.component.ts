import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ReportsService } from './reports.service';
import { 
  OperationalMetrics, 
  UtilizationReport, 
  FinancialReport, 
  AuditTrail 
} from './reports.model';
import { DropdownComponent } from '../../ui-library/Dropdown.component';
import { Heading1Component, SubtitleComponent } from '../../ui-library/Typography/Typography.component';
import { OperationalMetricsComponent } from './operational-metrics/operational-metrics.component';
import { UtilizationReportsComponent } from './utilization-reports/utilization-reports.component';
import { FinancialReportsComponent } from './financial-reports/financial-reports.component';
import { AuditTrailsComponent } from './audit-trails/audit-trails.component';

@Component({
  selector: 'app-reports',
  standalone: true,
  imports: [
    CommonModule, 
    FormsModule, 
    DropdownComponent, 
    Heading1Component, 
    SubtitleComponent,
    OperationalMetricsComponent,
    UtilizationReportsComponent,
    FinancialReportsComponent,
    AuditTrailsComponent
  ],
  template: `
    <div class="space-y-6">
      <!-- Header -->
      <div>
        <ui-heading1>Reports & Analytics</ui-heading1>
        <ui-subtitle>View warehouse performance reports and analytics</ui-subtitle>
      </div>

      <!-- Report Type Tabs -->
      <div class="card">
        <div class="border-b border-gray-200 dark:border-dark-700">
          <nav class="-mb-px flex space-x-8 px-6">
            @for (tab of reportTabs; track tab.id) {
              <button (click)="activeTab = tab.id"
                      [class]="getTabClass(tab.id)"
                      class="py-4 px-1 border-b-2 font-medium text-sm transition-colors">
                <svg class="h-5 w-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path [attr.d]="tab.icon" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" />
                </svg>
                {{ tab.name }}
              </button>
            }
          </nav>
        </div>

        <!-- Operational Metrics Tab -->
        @if (activeTab === 'operational') {
          <app-operational-metrics 
            [metrics]="operationalMetrics" 
            [period]="operationalPeriod"
            (periodChange)="operationalPeriod = $event; loadOperationalMetrics()"
            (export)="exportReport('operational')"
            />
        }

        <!-- Utilization Reports Tab -->
        @if (activeTab === 'utilization') {
          <app-utilization-reports
            [report]="utilizationReport"
            [period]="utilizationPeriod"
            (periodChange)="utilizationPeriod = $event; loadUtilizationReports()"
            (export)="exportReport('utilization')"
          />
        }

        <!-- Financial Reports Tab -->
        @if (activeTab === 'financial') {
          <app-financial-reports
            [report]="financialReport"
            [period]="financialPeriod"
            (periodChange)="financialPeriod = $event; loadFinancialReports()"
            (export)="exportReport('financial')"
          />
        }

        <!-- Audit Trails Tab -->
        @if (activeTab === 'audit') {
          <app-audit-trails
            [trails]="auditTrails"
            [filter]="auditFilter"
            [dateFrom]="auditDateFrom"
            [dateTo]="auditDateTo"
            (filterChange)="auditFilter = $event; loadAuditTrails()"
            (dateFromChange)="auditDateFrom = $event; loadAuditTrails()"
            (dateToChange)="auditDateTo = $event; loadAuditTrails()"
            (export)="exportReport('audit')"
          />
        }
      </div>
    </div>
  `
})
export class ReportsComponent implements OnInit {
  activeTab = 'operational';
  
  // Periods
  operationalPeriod = 'week';
  utilizationPeriod = 'month';
  financialPeriod = 'month';
  
  // Audit filters
  auditFilter = 'all';
  auditDateFrom = '';
  auditDateTo = '';

  // Data
  operationalMetrics: OperationalMetrics | null = null;
  utilizationReport: UtilizationReport | null = null;
  financialReport: FinancialReport | null = null;
  auditTrails: AuditTrail | null = null;

  reportTabs = [
    {
      id: 'operational',
      name: 'Operational Metrics',
      icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'
    },
    {
      id: 'utilization',
      name: 'Utilization Reports',
      icon: 'M4 6h16M4 10h16M4 14h16M4 18h16'
    },
    {
      id: 'financial',
      name: 'Financial Reports',
      icon: 'M13 10V3L4 14h7v7l9-11h-7z'
    },
    {
      id: 'audit',
      name: 'Audit Trails',
      icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z'
    }
  ];

  private reportsService = inject(ReportsService);

  ngOnInit(): void {
    this.loadOperationalMetrics();
    this.loadUtilizationReports();
    this.loadFinancialReports();
    this.loadAuditTrails();
  }

  loadOperationalMetrics(): void {
    this.reportsService.getOperationalMetrics(this.operationalPeriod).subscribe(metrics => {
      this.operationalMetrics = metrics;
    });
  }

  loadUtilizationReports(): void {
    this.reportsService.getUtilizationReport(this.utilizationPeriod).subscribe(report => {
      this.utilizationReport = report;
    });
  }

  loadFinancialReports(): void {
    this.reportsService.getFinancialReport(this.financialPeriod).subscribe(report => {
      this.financialReport = report;
    });
  }

  loadAuditTrails(): void {
    this.reportsService.getAuditTrails(this.auditFilter, this.auditDateFrom, this.auditDateTo).subscribe(trails => {
      this.auditTrails = trails;
    });
  }

  exportReport(type: string): void {
    this.reportsService.exportReport(type, this.getReportPeriod(type)).subscribe(blob => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${type}-report-${new Date().toISOString().split('T')[0]}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
    });
  }

  getReportPeriod(type: string): string {
    switch (type) {
      case 'operational': return this.operationalPeriod;
      case 'utilization': return this.utilizationPeriod;
      case 'financial': return this.financialPeriod;
      default: return 'month';
    }
  }

  getTabClass(tabId: string): string {
    return tabId === this.activeTab
      ? 'border-primary-500 text-primary-600'
      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300';
  }

}