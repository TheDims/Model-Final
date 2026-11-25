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