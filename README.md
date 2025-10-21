# Leukemia Federated Learning Project

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-khadijaparwez%2Flukemia--project-blue)](https://hub.docker.com/r/khadijaparwez/lukemia-project)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/KhadijaParwez/fl_xai_v1.3_29sept)

This project implements federated learning for leukemia detection using medical imaging data. The project includes machine learning models, data preprocessing utilities, and federated learning frameworks.


## 🐳 Docker Setup

### Option 1: Pull Existing Docker Container (Recommended)

If you want to quickly get started with the pre-built container:

```bash
# Option 1: Run without volume mount (simplest)
docker run -d \
  --name lukemia_project \
  -p 8888:8888 \
  khadijaparwez/lukemia-project:latest

# Option 2: Run with volume mount (for data persistence)
docker run -d \
  --name lukemia_project \
  -p 8888:8888 \
  -v $(pwd):/Volumes/mydata/projects/lukemia \
  khadijaparwez/lukemia-project:latest

# Access Jupyter Notebook
# Open your browser and go to: http://localhost:8888
```

**Note**: The Docker image is public, so no authentication is required to pull it.

### Option 2: Build Docker Container from Scratch

If you want to build the container locally or make modifications:

#### Prerequisites
- Docker installed on your system
- Docker Compose (optional, but recommended)

#### Build and Run with Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/KhadijaParwez/fl_xai_v1.3_29sept.git
cd fl_xai_v1.3_29sept

# Build the container
docker-compose build

# Run the container
docker-compose up -d

# Check if container is running
docker-compose ps

# View logs
docker-compose logs -f
```

#### Build and Run with Docker Commands

```bash
# Build the image
docker build -t lukemia-project .

# Run the container
docker run -d \
  --name lukemia_project \
  -p 8888:8888 \
  -v $(pwd):/Volumes/mydata/projects/lukemia \
  lukemia-project

# Check container status
docker ps

# View logs
docker logs -f lukemia_project
```

## 📁 Project Structure

```
lukemia/
├── content/
│   ├── datasets/
│   │   └── C-NMC_Leukemia/          # Leukemia dataset
│   └── drive/
│       └── MyDrive/
│           └── FL_Project/          # Federated learning configuration
├── models/                          # Trained models
├── dataset_utils.py                 # Dataset utilities
├── federated_learning_utils.py      # Federated learning utilities
├── modeling_utils.py                # Model utilities
├── training_utils.py                # Training utilities
├── xai_utils.py                     # Explainable AI utilities
├── plotting_utils.py                # Visualization utilities
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
└── XAI_Leukemia_14_July_v1.2.ipynb # Main Jupyter notebook
```

## 🔗 Repository Links

- **GitHub Repository**: [https://github.com/KhadijaParwez/fl_xai_v1.3_29sept](https://github.com/KhadijaParwez/fl_xai_v1.3_29sept)
- **Docker Hub**: [https://hub.docker.com/r/khadijaparwez/lukemia-project](https://hub.docker.com/r/khadijaparwez/lukemia-project)

## 🚀 Getting Started

1. **Pull or build the Docker container** using one of the methods above
2. **Access Jupyter Notebook** at `http://localhost:8888`
3. **Open the main notebook**: `XAI_Leukemia_14_July_v1.2.ipynb`
4. **Explore the dataset** in the `content/datasets/C-NMC_Leukemia/` directory

## 📊 Dataset

The project uses the C-NMC Leukemia dataset which includes:
- **Training data**: 3 folds with healthy (hem) and unhealthy (all) blood cell images
- **Validation data**: Preliminary test data with labels
- **Testing data**: Final test data for evaluation

## 🔧 Dependencies

Key dependencies include:
- Python 3.8
- TensorFlow
- Keras
- Jupyter Notebook
- tf-keras-vis (for explainable AI)
- Various ML and data processing libraries

See `requirements.txt` for the complete list of dependencies.

## 🛠️ Development

### Adding New Features

1. Make your changes to the source code
2. Rebuild the container:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

### Accessing Container Shell

```bash
# Access running container
docker exec -it lukemia_project /bin/bash

# Or with docker-compose
docker-compose exec lukemia-notebook /bin/bash
```

## 📝 Usage

### Federated Learning
- Use `federated_learning_utils.py` for federated learning implementations
- Configuration files are in `content/drive/MyDrive/FL_Project/conf/`

### Model Training
- Use `training_utils.py` for training utilities
- Use `modeling_utils.py` for model definitions

### Explainable AI
- Use `xai_utils.py` for explainable AI implementations
- Main notebook includes XAI demonstrations

## 🐛 Troubleshooting

### Container Issues
```bash
# Stop and remove container
docker-compose down

# Remove all containers and images
docker-compose down --rmi all

# Rebuild from scratch
docker-compose build --no-cache
```

### Port Conflicts
If port 8888 is already in use, modify the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8889:8888"  # Use port 8889 instead
```

### Permission Issues
On Linux/macOS, you might need to adjust file permissions:
```bash
sudo chown -R $USER:$USER .
```

## 📄 License

MIT License

Copyright (c) 2025 KhadijaParwez

---

**Note**: This project is for research and educational purposes. Please ensure you have appropriate permissions and follow ethical guidelines when working with medical data.
