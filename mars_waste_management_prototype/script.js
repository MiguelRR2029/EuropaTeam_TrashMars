// ========================================
// Data Models
// ========================================

const materialsData = [
  { name: 'Zotek F30', quantity: 1000, purity: 100, applications: 'Empaques y aislamiento de carga útil' },
  { name: 'Cargo Transfer Bags (Nomex)', quantity: 100, purity: 92, applications: 'Zonas de actividad extravehicular' },
  { name: 'Cargo Transfer Bags (Nylon)', quantity: 100, purity: 3, applications: 'Zonas de actividad extravehicular' },
  { name: 'Cargo Transfer Bags (Polyester)', quantity: 100, purity: 2, applications: 'Zonas de actividad extravehicular' },
  { name: 'Clothing (Cotton)', quantity: 770, purity: 56, applications: 'Zonas de Aterrizaje' },
  { name: 'Clothing (Nylon)', quantity: 770, purity: 6, applications: 'Zonas de Aterrizaje' },
  { name: 'Clothing (Polyester)', quantity: 770, purity: 38, applications: 'Zonas de Aterrizaje' },
  { name: 'Towels/Wash Cloths', quantity: 210, purity: 100, applications: 'Zonas de Aterrizaje' },
  { name: 'Cleaning Wipes', quantity: 20, purity: 100, applications: 'Zonas de Aterrizaje' },
  { name: 'Overwrap (Polyester)', quantity: 290, purity: 13, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Overwrap (Polyethylene)', quantity: 290, purity: 15, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Overwrap (Aluminum)', quantity: 290, purity: 30, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Rehydratable Pouch (Nylon)', quantity: 390, purity: 41, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Rehydratable Pouch (Polyethylene)', quantity: 390, purity: 33, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Rehydratable Pouch (EVOH)', quantity: 390, purity: 11, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Drink Pouch (Aluminum)', quantity: 80, purity: 24, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Drink Pouch (Polyethylene)', quantity: 80, purity: 65, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Drink Pouch (PET)', quantity: 80, purity: 11, applications: 'Módulos de Aterrizaje/Rovers' },
  { name: 'Aluminum Structure/Struts', quantity: 900, purity: 90, applications: 'Estructuras de Descenso' },
  { name: 'Polymer Matrix Composites (Thermoset)', quantity: 100, purity: 40, applications: 'Estructuras de Descenso' },
  { name: 'Polymer Matrix Composites (Carbon Fiber)', quantity: 100, purity: 60, applications: 'Estructuras de Descenso' },
  { name: 'Air cushion', quantity: 4, purity: 100, applications: 'Basura dispersa cerca de los sitios de aterrizaje' },
  { name: 'Bubble wrap filler', quantity: 1, purity: 100, applications: 'Basura dispersa cerca de los sitios de aterrizaje' },
  { name: 'Reclosable bags', quantity: 9, purity: 100, applications: 'Basura dispersa cerca de los sitios de aterrizaje' },
  { name: 'Anti-Static Bubble Wrap Bags', quantity: 9, purity: 100, applications: 'Basura dispersa cerca de los sitios de aterrizaje' },
  { name: 'Plastazote', quantity: 36, purity: 100, applications: 'Basura dispersa cerca de los sitios de aterrizaje' },
  { name: 'Nitrile gloves', quantity: 41, purity: 100, applications: 'Basura dispersa cerca de los sitios de aterrizaje' }
];


const wastePoints = [
    { id: 'WP-01', name: 'Hab. Principal', x: 250, y: 200, level: 'high', amount: 342 },
    { id: 'WP-02', name: 'Lab. Científico', x: 450, y: 180, level: 'medium', amount: 187 },
    { id: 'WP-03', name: 'Invernadero', x: 350, y: 320, level: 'low', amount: 89 },
    { id: 'WP-04', name: 'Taller Mecánico', x: 150, y: 350, level: 'high', amount: 298 },
    { id: 'WP-05', name: 'Centro Médico', x: 550, y: 280, level: 'low', amount: 67 },
    { id: 'WP-06', name: 'Almacén Central', x: 350, y: 150, level: 'medium', amount: 156 },
    { id: 'WP-07', name: 'Comedor', x: 200, y: 280, level: 'medium', amount: 178 },
    { id: 'WP-08', name: 'Gimnasio', x: 500, y: 380, level: 'low', amount: 54 },
    { id: 'WP-09', name: 'Centro Comando', x: 350, y: 240, level: 'low', amount: 43 },
    { id: 'WP-10', name: 'Planta Energía', x: 100, y: 150, level: 'medium', amount: 134 },
    { id: 'WP-11', name: 'Hangar Vehículos', x: 550, y: 100, level: 'high', amount: 412 },
    { id: 'WP-12', name: 'Zona Recreativa', x: 450, y: 420, level: 'low', amount: 76 }
];

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    updateLastUpdateTime();
    initializeMarsMap();
    initializeMaterialsChart();
    initializeSankeyChart();
    initializeInventoryTable();
    initializeAssignmentForm();
    
    // Update time every minute
    setInterval(updateLastUpdateTime, 60000);
});

// ========================================
// Time Update
// ========================================

function updateLastUpdateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('es-ES', { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
    });
    document.getElementById('lastUpdate').textContent = timeString;
}

// ========================================
// Mars Base Map
// ========================================

function initializeMarsMap() {
    const container = document.getElementById('marsBaseMap');
    
    // Create SVG
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('id', 'marsBaseMapSVG');
    svg.setAttribute('viewBox', '0 0 650 500');
    
    // Add background pattern (Mars terrain)
    addMarsTerrainPattern(svg);
    
    // Add base structures
    addBaseStructures(svg);
    
    // Add connection lines
    addConnectionLines(svg);
    
    // Add waste collection points
    addWastePoints(svg);
    
    // Add tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'map-tooltip';
    tooltip.id = 'mapTooltip';
    container.appendChild(tooltip);
    
    container.appendChild(svg);
}

function addMarsTerrainPattern(svg) {
    // Add some decorative circles to simulate craters
    const craters = [
        { cx: 100, cy: 80, r: 30 },
        { cx: 550, cy: 450, r: 25 },
        { cx: 50, cy: 400, r: 20 },
        { cx: 600, cy: 50, r: 15 }
    ];
    
    craters.forEach(crater => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', crater.cx);
        circle.setAttribute('cy', crater.cy);
        circle.setAttribute('r', crater.r);
        circle.setAttribute('fill', 'rgba(0, 0, 0, 0.1)');
        circle.setAttribute('stroke', 'rgba(0, 0, 0, 0.2)');
        circle.setAttribute('stroke-width', '1');
        svg.appendChild(circle);
    });
}

function addBaseStructures(svg) {
    const structures = [
        { x: 200, y: 150, width: 300, height: 200, label: 'COMPLEJO CENTRAL' },
        { x: 100, y: 100, width: 80, height: 80, label: 'Módulo A' },
        { x: 520, y: 50, width: 80, height: 80, label: 'Módulo B' },
        { x: 520, y: 250, width: 80, height: 80, label: 'Módulo C' },
        { x: 100, y: 300, width: 80, height: 80, label: 'Módulo D' },
        { x: 420, y: 370, width: 80, height: 80, label: 'Módulo E' }
    ];
    
    structures.forEach(struct => {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', struct.x);
        rect.setAttribute('y', struct.y);
        rect.setAttribute('width', struct.width);
        rect.setAttribute('height', struct.height);
        rect.setAttribute('rx', '10');
        rect.setAttribute('class', 'base-structure');
        svg.appendChild(rect);
        
        // Add label
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', struct.x + struct.width / 2);
        text.setAttribute('y', struct.y + struct.height / 2);
        text.setAttribute('class', 'map-label');
        text.setAttribute('dominant-baseline', 'middle');
        text.textContent = struct.label;
        svg.appendChild(text);
    });
}

function addConnectionLines(svg) {
    const connections = [
        { x1: 180, y1: 140, x2: 250, y2: 200 },
        { x1: 520, y1: 130, x2: 450, y2: 180 },
        { x1: 560, y1: 250, x2: 500, y2: 250 },
        { x1: 180, y1: 340, x2: 250, y2: 320 },
        { x1: 460, y1: 370, x2: 420, y2: 350 }
    ];
    
    connections.forEach(conn => {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', conn.x1);
        line.setAttribute('y1', conn.y1);
        line.setAttribute('x2', conn.x2);
        line.setAttribute('y2', conn.y2);
        line.setAttribute('class', 'connection-line');
        svg.appendChild(line);
    });
}

function addWastePoints(svg) {
    wastePoints.forEach(point => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', point.x);
        circle.setAttribute('cy', point.y);
        circle.setAttribute('r', '8');
        circle.setAttribute('class', `waste-point waste-point-${point.level}`);
        circle.setAttribute('data-id', point.id);
        circle.setAttribute('data-name', point.name);
        circle.setAttribute('data-level', point.level);
        circle.setAttribute('data-amount', point.amount);
        
        // Add event listeners
        circle.addEventListener('mouseenter', showTooltip);
        circle.addEventListener('mousemove', moveTooltip);
        circle.addEventListener('mouseleave', hideTooltip);
        
        svg.appendChild(circle);
    });
}

function showTooltip(event) {
    const tooltip = document.getElementById('mapTooltip');
    const id = event.target.getAttribute('data-id');
    const name = event.target.getAttribute('data-name');
    const level = event.target.getAttribute('data-level');
    const amount = event.target.getAttribute('data-amount');
    
    const levelText = {
        'high': 'Alta',
        'medium': 'Media',
        'low': 'Baja'
    };
    
    tooltip.innerHTML = `
        <strong>${id}</strong><br>
        ${name}<br>
        Acumulación: ${levelText[level]}<br>
        Cantidad: ${amount} kg
    `;
    
    tooltip.classList.add('show');
    moveTooltip(event);
}

function moveTooltip(event) {
    const tooltip = document.getElementById('mapTooltip');
    const container = document.getElementById('marsBaseMap');
    const rect = container.getBoundingClientRect();
    
    tooltip.style.left = (event.clientX - rect.left + 15) + 'px';
    tooltip.style.top = (event.clientY - rect.top + 15) + 'px';
}

function hideTooltip() {
    const tooltip = document.getElementById('mapTooltip');
    tooltip.classList.remove('show');
}

// ========================================
// Materials Chart (Highcharts)
// ========================================

function initializeMaterialsChart() {
    Highcharts.chart('materialsChart', {
        chart: {
            type: 'column',
            backgroundColor: 'transparent'
        },
        title: {
            text: 'Cantidad de Materiales Reciclados',
            style: {
                color: '#2d3748',
                fontSize: '16px',
                fontWeight: 'bold'
            }
        },
        xAxis: {
            categories: materialsData.map(m => m.name),
            labels: {
                rotation: -45,
                style: {
                    fontSize: '11px'
                }
            }
        },
        yAxis: {
            min: 0,
            title: {
                text: 'Cantidad (kg)'
            }
        },
        legend: {
            enabled: false
        },
        tooltip: {
            headerFormat: '<b>{point.x}</b><br/>',
            pointFormat: 'Cantidad: {point.y} kg<br/>Pureza: {point.purity}%<br/>Aplicaciones: {point.applications}'
        },
        plotOptions: {
            column: {
                colorByPoint: true,
                dataLabels: {
                    enabled: true,
                    format: '{point.y} kg'
                }
            }
        },
        colors: ['#667eea', '#56ab2f', '#f093fb', '#fa709a', '#c1440e', '#764ba2', '#a8e063', '#f5576c'],
        series: [{
            name: 'Materiales',
            data: materialsData.map(m => ({
                y: m.quantity,
                purity: m.purity,
                applications: m.applications
            }))
        }],
        credits: {
            enabled: false
        }
    });
}

// ========================================
// Sankey Chart (Highcharts)
// ========================================

function initializeSankeyChart() {
    Highcharts.chart('sankeyChart', {
        chart: {
            backgroundColor: 'transparent',
            height: 500
        },
        title: {
            text: 'Flujo del Ciclo de Reciclaje',
            style: {
                color: '#2d3748',
                fontSize: '16px',
                fontWeight: 'bold'
            }
        },
        accessibility: {
            point: {
                valueDescriptionFormat: '{index}. {point.from} to {point.to}, {point.weight}.'
            }
        },
        tooltip: {
            headerFormat: null,
            pointFormat: '{point.fromNode.name} → {point.toNode.name}: <b>{point.weight} kg</b>',
            nodeFormat: '{point.name}: <b>{point.sum} kg</b>'
        },
        series: [{
            keys: ['from', 'to', 'weight'],
            data: [
                // Recolección inicial
                ['Residuos Generados', 'Plástico PET', 342],
                ['Residuos Generados', 'Aluminio', 187],
                ['Residuos Generados', 'Acero', 523],
                ['Residuos Generados', 'Vidrio', 156],
                ['Residuos Generados', 'Cobre', 89],
                ['Residuos Generados', 'Titanio', 124],
                ['Residuos Generados', 'Compuestos Org.', 278],
                ['Residuos Generados', 'Polímeros Avanz.', 224],
                
                // Procesamiento
                ['Plástico PET', 'Construcción', 200],
                ['Plástico PET', 'Contenedores', 142],
                ['Aluminio', 'Estructuras', 120],
                ['Aluminio', 'Herramientas', 67],
                ['Acero', 'Construcción', 350],
                ['Acero', 'Componentes', 173],
                ['Vidrio', 'Ventanas', 100],
                ['Vidrio', 'Contenedores', 56],
                ['Cobre', 'Electrónica', 55],
                ['Cobre', 'Cables', 34],
                ['Titanio', 'Estructuras', 80],
                ['Titanio', 'Herramientas', 44],
                ['Compuestos Org.', 'Fertilizantes', 180],
                ['Compuestos Org.', 'Biogás', 98],
                ['Polímeros Avanz.', 'Componentes', 150],
                ['Polímeros Avanz.', 'Sellantes', 74],
                
                // Aplicaciones finales
                ['Construcción', 'Expansión Base', 550],
                ['Estructuras', 'Expansión Base', 200],
                ['Herramientas', 'Operaciones', 111],
                ['Componentes', 'Mantenimiento', 323],
                ['Contenedores', 'Almacenamiento', 198],
                ['Ventanas', 'Expansión Base', 100],
                ['Electrónica', 'Sistemas', 55],
                ['Cables', 'Sistemas', 34],
                ['Fertilizantes', 'Agricultura', 180],
                ['Biogás', 'Energía', 98],
                ['Sellantes', 'Mantenimiento', 74]
            ],
            type: 'sankey',
            name: 'Flujo de Materiales'
        }],
        credits: {
            enabled: false
        }
    });
}

// ========================================
// Inventory Table
// ========================================

function initializeInventoryTable() {
    const tbody = document.querySelector('#inventoryTable tbody');
    
    materialsData.forEach(material => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${material.name}</strong></td>
            <td>${material.quantity}</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${getPurityClass(material.purity)}" 
                         role="progressbar" 
                         style="width: ${material.purity}%"
                         aria-valuenow="${material.purity}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                        ${material.purity}%
                    </div>
                </div>
            </td>
            <td>${material.applications}</td>
            <td>
                <button class="btn btn-sm btn-primary" onclick="selectMaterial('${material.name}', ${material.quantity})">
                    <i class="bi bi-arrow-right-circle"></i> Asignar
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function getPurityClass(purity) {
    if (purity >= 95) return 'bg-success';
    if (purity >= 90) return 'bg-info';
    if (purity >= 85) return 'bg-warning';
    return 'bg-danger';
}

// ========================================
// Assignment Form
// ========================================

function initializeAssignmentForm() {
    const select = document.getElementById('materialSelect');
    
    materialsData.forEach(material => {
        const option = document.createElement('option');
        option.value = material.name;
        option.textContent = `${material.name} (${material.quantity} kg disponibles)`;
        option.setAttribute('data-quantity', material.quantity);
        select.appendChild(option);
    });
    
    // Event listener for material selection
    select.addEventListener('change', function() {
        const selectedOption = this.options[this.selectedIndex];
        const quantity = selectedOption.getAttribute('data-quantity') || 0;
        document.getElementById('availableQuantity').textContent = quantity;
        document.getElementById('quantityInput').setAttribute('max', quantity);
    });
    
    // Form submission
    document.getElementById('assignmentForm').addEventListener('submit', function(e) {
        e.preventDefault();
        handleAssignment();
    });
}

function selectMaterial(name, quantity) {
    const select = document.getElementById('materialSelect');
    select.value = name;
    document.getElementById('availableQuantity').textContent = quantity;
    document.getElementById('quantityInput').setAttribute('max', quantity);
    document.getElementById('quantityInput').focus();
    
    // Scroll to assignment form
    document.getElementById('assignmentForm').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function handleAssignment() {
    const material = document.getElementById('materialSelect').value;
    const quantity = parseFloat(document.getElementById('quantityInput').value);
    const purpose = document.getElementById('purposeSelect').value;
    const notes = document.getElementById('notesInput').value;
    const available = parseFloat(document.getElementById('availableQuantity').textContent);
    
    const alertDiv = document.getElementById('assignmentAlert');
    
    if (quantity > available) {
        alertDiv.className = 'alert alert-danger mt-3';
        alertDiv.innerHTML = `
            <i class="bi bi-exclamation-triangle-fill me-2"></i>
            <strong>Error:</strong> La cantidad solicitada (${quantity} kg) excede la cantidad disponible (${available} kg).
        `;
        alertDiv.style.display = 'block';
        return;
    }
    
    // Success message
    alertDiv.className = 'alert alert-success mt-3';
    alertDiv.innerHTML = `
        <i class="bi bi-check-circle-fill me-2"></i>
        <strong>¡Asignación exitosa!</strong><br>
        Material: ${material}<br>
        Cantidad: ${quantity} kg<br>
        Propósito: ${document.getElementById('purposeSelect').options[document.getElementById('purposeSelect').selectedIndex].text}
        ${notes ? '<br>Notas: ' + notes : ''}
    `;
    alertDiv.style.display = 'block';
    
    // Reset form
    document.getElementById('assignmentForm').reset();
    document.getElementById('availableQuantity').textContent = '0';
    
    // Hide alert after 5 seconds
    setTimeout(() => {
        alertDiv.style.display = 'none';
    }, 5000);
}
