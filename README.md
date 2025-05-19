# Ising Visualizer Web App

This Streamlit app simulates a 2D Ising model using the Wolff algorithm and visualizes spin configurations and phase transition behavior.

## Features
- Adjustable lattice size and temperature range
- Simulation of Ising model dynamics using cluster updates
- Plotting magnetization and magnetic susceptibility
- Arrow visualization of spin states
- Downloadable snapshots

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app locally
```bash
streamlit run app.py
```

### 3. Parameters
You can modify:
- Lattice size (L)
- Temperature range (Tmin to Tmax)
- Number of Monte Carlo trials per temperature

### 4. Deploy on Streamlit Cloud
1. Push all files to a public GitHub repository
2. Go to https://streamlit.io/cloud
3. Click "New App"
4. Choose your repo and set `app.py` as the main file

## Files
- `app.py`: Main Streamlit interface
- `ising_model.py`: Wolff algorithm and magnetization/susceptibility calculations
- `visualizer.py`: Spin configuration arrow plot function
- `requirements.txt`: Required Python packages

## Example Output
- Arrow plot showing spins as ↑ and ↓
- Dual-axis plot of magnetization and susceptibility
- Interactive snapshot selection and image download

## Author
This project was developed by Yuchen Xiong, an independent researcher and developer interested in statistical physics and computational modeling.
