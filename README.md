# Algorithmic Trading System - Gold (XAU/USD)

A Python-based algorithmic trading system that uses technical analysis to identify trading signals for gold (XAU/USD) using candlestick patterns, support/resistance levels, and reversal signals.

## Features

- **Real-time Data**: Fetches live gold price data using Twelve Data API
- **Technical Analysis**: Implements multiple technical indicators:
  - Support and Resistance level detection
  - Engulfing candlestick pattern recognition
  - Star pattern (hammer/shooting star) detection
  - Price action analysis
- **Signal Generation**: Combines multiple technical indicators to generate buy/sell signals
- **Visualization**: Interactive candlestick charts with support/resistance levels and trading signals
- **Configurable Parameters**: Adjustable thresholds and timeframes for different trading strategies

## Prerequisites

- Python 3.7+
- Twelve Data API key (free tier available)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd randomforesttrading
```

2. Install required dependencies:
```bash
pip install numpy pandas mplfinance python-dotenv twelvedata
```

3. Set up your API key:
   - Create a `secrets.env` file in the project root
   - Add your Twelve Data API key:
   ```
   TD_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

Run the main trading script:
```bash
python auto.py
```

### Configuration

The script can be configured by modifying these parameters in `auto.py`:

- **Symbol**: Currently set to "XAU/USD" (gold)
- **Interval**: Timeframe for data (currently 15min)
- **Output Size**: Number of data points (currently 500)
- **Support/Resistance Parameters**:
  - `n1`, `n2`: Lookback periods for level detection
  - `threshold_ratio`: Minimum distance between levels
- **Signal Parameters**:
  - `bodydiffmin`: Minimum body size for pattern recognition
  - `backCandles`: Number of candles to analyze

### Trading Signals

The system generates three types of signals:

1. **Signal 1**: Engulfing pattern near resistance levels (bearish)
2. **Signal 2**: Engulfing pattern near support levels (bullish)
3. **Star Patterns**: Hammer/shooting star patterns for additional confirmation

## Technical Indicators

### Support and Resistance Detection
- Uses swing high/low analysis
- Filters nearby levels to avoid redundancy
- Configurable sensitivity parameters

### Candlestick Patterns
- **Engulfing Patterns**: Bullish and bearish engulfing candles
- **Star Patterns**: Hammer and shooting star formations
- **Body Size Analysis**: Considers candlestick body-to-wick ratios

### Signal Confirmation
- Combines pattern recognition with support/resistance proximity
- Uses multiple timeframe analysis
- Implements risk management through level-based entries

## Data Sources

- **Primary**: Twelve Data API for real-time gold price data
- **Backup**: Local CSV files for historical analysis
- **Timeframe**: 15-minute intervals (configurable)

## Visualization

The system generates interactive candlestick charts showing:
- Price action with candlestick patterns
- Support and resistance levels (green/red lines)
- Trading signals (buy/sell markers)
- Technical indicator overlays

## Risk Disclaimer

⚠️ **Important**: This is for educational and research purposes only. Algorithmic trading involves significant risk of loss. Always:
- Test strategies thoroughly before live trading
- Use proper risk management
- Never invest more than you can afford to lose
- Consider consulting with financial advisors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with your local financial regulations before using for actual trading.

## Support

For issues and questions:
- Check the code comments for parameter explanations
- Review the Twelve Data API documentation
- Ensure your API key has sufficient permissions

## Future Enhancements

- [ ] Machine learning integration (Random Forest as suggested by project name)
- [ ] Backtesting framework
- [ ] Risk management module
- [ ] Multiple asset support
- [ ] Web interface
- [ ] Real-time alerts
- [ ] Performance analytics