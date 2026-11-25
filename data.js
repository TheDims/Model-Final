const power = [0,0,0,0,0,0,0.1,0.4,0.9,1.3,1.6,1.9,2.3,2.1,1.9,1.5,1.0,0.5,0.1,0,0,0,0,0,0]; // 0~24시, 1시간 간격 (kW)

const DATA = {
  greeting: 'Namaste!',
  badge: 'demo · 2020 data',
  location: 'Andhra Pradesh, India',
  date: 'Sun, 14 Jun 2020',
  temp: '33°C',
  condition: 'Cloudy',

  today: {
    kwh: 15.6,
    changePct: -3,
    rangeLow: 9.5,
    rangeHigh: 20.9,
    rupees: 109,
    peakHour: 12,
    peakKw: 2.3,
    power: power,
    upper: power.map(v => +(v * 1.3).toFixed(2)),
    lower: power.map(v => +(v * 0.6).toFixed(2))
  },

  consumptionKwh: 10.0,
  action: {
    title: 'Charge',
    time: 'between 10:45 AM – 2:45 PM',
    note: 'Best time to charge your inverter battery or use extra solar.'
  },
  weather: { title: 'Cloudy', detail: 'High 33°C | Low 25°C · clouds 96%' },
  plan: 'About 6 kWh more than you use. Run the water pump, washing machine or geyser in the sunny hours.',
  nextDays: [
    { day: 'Mon', date: '15 Jun', icon: '☁️', kwh: 15.0 },
    { day: 'Tue', date: '16 Jun', icon: '☁️', kwh: 13.2 },
    { day: 'Wed', date: '17 Jun', icon: '☁️', kwh: 13.5 }
  ],
  env: { co2: 10.9, note: '≈ what 0.5 trees absorb in a year' },
  footnote: 'Demo with 2020 data (Plant 1, location estimated at 14.5°N, 77.5°E, Andhra Pradesh). Times in IST. Money saved uses ₹7.0/kWh (assumption). Weather input is ERA5 (weather measured afterwards); a real app uses a weather forecast, so real errors are a bit larger.'
};

const FORECAST = {
  selected: 'Sun 14',
  description: 'Forecast for each day of the demo period (29 May – 17 June 2020). Each day was forecast by a model that only saw earlier days. The measured value is shown to check the model.',
  days: [
    { label: 'Fri 29', forecast: 17.0, measured: 18.5 },
    { label: 'Sat 30', forecast: 16.8, measured: 16.4 },
    { label: 'Sun 31', forecast: 15.5, measured: 13.6 },
    { label: 'Mon 1',  forecast: 16.5, measured: 13.1 },
    { label: 'Tue 2',  forecast: 16.6, measured: 16.8 },
    { label: 'Wed 3',  forecast: 14.2, measured: 17.0 },
    { label: 'Thu 4',  forecast: 15.7, measured: 18.5 },
    { label: 'Fri 5',  forecast: 16.6, measured: 17.4 },
    { label: 'Sat 6',  forecast: 16.5, measured: 14.4 },
    { label: 'Sun 7',  forecast: 16.6, measured: 18.3 },
    { label: 'Mon 8',  forecast: 17.4, measured: 18.9 },
    { label: 'Tue 9',  forecast: 16.5, measured: 18.6 },
    { label: 'Wed 10', forecast: 13.4, measured: 14.6 },
    { label: 'Thu 11', forecast: 12.0, measured: 12.7 },
    { label: 'Fri 12', forecast: 12.8, measured: 13.2 },
    { label: 'Sat 13', forecast: 16.1, measured: 19.2 },
    { label: 'Sun 14', forecast: 15.6, measured: 16.6 },
    { label: 'Mon 15', forecast: 15.0, measured: 15.3 },
    { label: 'Tue 16', forecast: 13.2, measured: 14.1 },
    { label: 'Wed 17', forecast: 13.5, measured: 13.4 }
  ]
};