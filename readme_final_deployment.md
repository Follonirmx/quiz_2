# 🤖 Despliegue de PyBullet Industrial Robotics Gym en Docker

## 📋 Descripción del Proyecto

Este documento describe el proceso completo de despliegue del proyecto **PyBullet Industrial Robotics Gym** en un contenedor Docker para Windows. El proyecto implementa algoritmos de Deep Reinforcement Learning (DRL) para planificación de movimientos en robots industriales utilizando PyBullet como motor de simulación física.

### 🎯 Objetivos

- Desplegar el environment E1 (sin obstáculos de colisión)
- Entrenar modelos DRL usando algoritmos TD3, SAC y DDPG
- Ejecutar simulaciones de robots industriales en un entorno containerizado
- Configurar el entorno para entrenamiento en CPU

---

## 🏗️ Arquitectura del Despliegue

```
┌─────────────────────────────────────┐
│         Windows Host                │
│  ┌───────────────────────────────┐  │
│  │   Docker Desktop (WSL 2)      │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │  Container Ubuntu/Debian│  │  │
│  │  │  - Python 3.9           │  │  │
│  │  │  - PyBullet             │  │  │
│  │  │  - PyTorch (CPU)        │  │  │
│  │  │  - Stable-Baselines3    │  │  │
│  │  │  - Gymnasium            │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
│                                     │
│  Volúmenes Persistentes:            │
│  └─ ./data/ → Modelos y resultados │
└─────────────────────────────────────┘
```

---

## 📦 Componentes del Despliegue

### 1. **Dockerfile**

Imagen base: `python:3.9-slim-bullseye`

**Dependencias del Sistema:**
- Git
- Build-essential
- Librerías OpenGL (libgl1-mesa-glx, libglib2.0-0, libsm6, etc.)

**Dependencias Python:**
- PyTorch 2.0.1 (versión CPU)
- PyBullet 3.2.5
- Stable-Baselines3 2.0.0
- Gymnasium 0.28.1
- Matplotlib, Pandas, SciPy, NumPy

**Características:**
- Clonación automática del repositorio
- Configuración de PYTHONPATH
- Instalación optimizada sin caché para reducir tamaño
- Imagen final: ~2-3 GB

### 2. **docker-compose.yml**

**Configuración:**
- Nombre del contenedor: `pybullet_industrial_robotics`
- Volúmenes persistentes para datos de entrenamiento
- Límites de recursos: 4 CPUs, 8GB RAM
- Modo interactivo (stdin_open + tty)
- Puerto 8888 expuesto para extensiones futuras

### 3. **Estructura de Datos**

```
proyecto/
├── Dockerfile
├── docker-compose.yml
├── data/                          # Volumen persistente
│   ├── Model/                     # Modelos entrenados
│   │   └── Environment_Default/
│   │       └── TD3/
│   │           └── Universal_Robots_UR3/
│   ├── Training/                  # Datos de entrenamiento
│   │   └── Environment_Default/
│   │       └── TD3/
│   │           └── Universal_Robots_UR3/
│   │               ├── progress.csv
│   │               ├── monitor.csv
│   │               └── time.txt
│   └── Prediction/                # Resultados de predicción
└── custom_scripts/                # Scripts personalizados
```

---

## 🚀 Proceso de Instalación

### Requisitos Previos

- **Sistema Operativo:** Windows 10/11 (64-bit)
- **Docker Desktop:** Versión 20.10 o superior con WSL 2
- **RAM:** Mínimo 8GB (16GB recomendado)
- **Espacio en Disco:** Mínimo 20GB libres
- **CPU:** 4 núcleos o más (recomendado)

### Paso 1: Instalación de Docker Desktop

1. Descargar Docker Desktop desde: https://www.docker.com/products/docker-desktop/
2. Instalar seleccionando "Use WSL 2 instead of Hyper-V"
3. Configurar recursos en Settings → Resources:
   - CPUs: 4
   - Memory: 8GB
   - Disk: 60GB

### Paso 2: Configuración del Proyecto

```powershell
# Crear estructura de directorios
mkdir C:\pybullet-project
cd C:\pybullet-project
mkdir data, custom_scripts

# Crear archivos de configuración
# - Dockerfile (contenido proporcionado)
# - docker-compose.yml (contenido proporcionado)
```

### Paso 3: Construcción de la Imagen

```powershell
# Construir la imagen Docker
docker-compose build --no-cache

# Tiempo estimado: 10-15 minutos
```

### Paso 4: Inicialización del Contenedor

```powershell
# Iniciar el contenedor en segundo plano
docker-compose up -d

# Verificar que está corriendo
docker ps
```

---

## ⚙️ Configuración del Environment E1

### Modificaciones en el Código

#### 1. Configuración de Device (CPU)

**Archivo:** `Training/train_td3.py`

**Cambios realizados:**
```python
# Línea 104-105 y 107-108
# ANTES: device='cuda'
# DESPUÉS: device='cpu'

model = stable_baselines3.TD3(
    policy="MultiInputPolicy", 
    env=gym_environment, 
    gamma=0.95, 
    learning_rate=0.001, 
    action_noise=action_noise, 
    device='cpu',  # ← Cambio aquí
    batch_size=256, 
    policy_kwargs=dict(net_arch=[256, 256, 256]), 
    verbose=1
)
```

#### 2. Configuración del Modo de Simulación

**Archivo:** `src/core.py`

**Cambios realizados:**
```python
# Para entrenamiento sin interfaz gráfica (recomendado para Docker)
p.connect(p.DIRECT)  # Modo headless

# Para entrenamiento con interfaz gráfica (requiere X server)
# p.connect(p.GUI, options="--width=1280 --height=720")
```

#### 3. Parámetros del Environment E1

**Configuración en `train_td3.py`:**
```python
# Tipo de robot
CONST_ROBOT_TYPE = Parameters.Universal_Robots_UR3_Str

# Modo de environment (E1 = sin obstáculos)
CONST_ENV_MODE = 'Default'

# Algoritmo de entrenamiento
CONST_ALGORITHM = 'TD3'

# Pasos de entrenamiento
total_timesteps = 100000  # Ajustable según necesidades
```

---

## 🎮 Ejecución del Entrenamiento

### Acceso al Contenedor

```powershell
# Entrar al contenedor
docker exec -it pybullet_industrial_robotics bash
```

### Verificación del Entorno

```bash
# Verificar instalación de paquetes
python -c "import pybullet; print('PyBullet OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import stable_baselines3; print('Stable-Baselines3 OK')"

# Verificar estructura del proyecto
ls -la /workspace/PyBullet_Industrial_Robotics_Gym/
```

### Entrenamiento con TD3

```bash
# Navegar al directorio de entrenamiento
cd /workspace/PyBullet_Industrial_Robotics_Gym/Training

# Ejecutar entrenamiento
python train_td3.py
```

### Entrenamiento con otros algoritmos

```bash
# SAC (Soft Actor-Critic)
python train_sac.py

# DDPG (Deep Deterministic Policy Gradient)
python train_ddpg.py
```

### Salida Esperada

```
[INFO] The file has been successfully removed.
[INFO] >> /workspace/.../progress.csv
[INFO] The calculation is in progress.
pybullet build time: May 20 2022 19:45:31
startThreads creating 1 threads.
starting thread 0
started thread 0
All functions dynamically loaded using dlopen/dlsym OK!
----------------------------------
| rollout/           |          |
|    ep_len_mean     | 50.0     |
|    ep_rew_mean     | -234.56  |
| time/              |          |
|    fps             | 125      |
|    iterations      | 100      |
----------------------------------
```

---

## 📊 Monitoreo y Resultados

### Monitoreo en Tiempo Real

```powershell
# Ver logs del contenedor
docker logs -f pybullet_industrial_robotics

# Ver estadísticas de recursos
docker stats pybullet_industrial_robotics
```

### Acceso a Resultados desde Windows

Los resultados se almacenan automáticamente en:

```
C:\pybullet-project\data\
├── Model\Environment_Default\TD3\Universal_Robots_UR3\
│   └── model.zip                    # Modelo entrenado
├── Training\Environment_Default\TD3\Universal_Robots_UR3\
│   ├── progress.csv                 # Progreso del entrenamiento
│   ├── monitor.csv                  # Métricas del environment
│   └── time.txt                     # Tiempo de entrenamiento
```

### Visualización de Resultados

```python
# Leer datos de entrenamiento
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('C:/pybullet-project/data/Training/.../progress.csv')

# Graficar recompensa promedio
plt.plot(data['time/total_timesteps'], data['rollout/ep_rew_mean'])
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward Mean')
plt.title('Training Progress')
plt.show()
```

---

## 🔧 Comandos Útiles de Docker

### Gestión del Contenedor

```powershell
# Iniciar contenedor
docker-compose up -d

# Detener contenedor
docker-compose down

# Reiniciar contenedor
docker-compose restart

# Ver estado
docker ps

# Ver logs
docker-compose logs -f

# Entrar al contenedor
docker exec -it pybullet_industrial_robotics bash

# Salir del contenedor (sin detenerlo)
exit  # o Ctrl+D
```

### Transferencia de Archivos

```powershell
# Copiar archivo de Windows a contenedor
docker cp C:\ruta\archivo.py pybullet_industrial_robotics:/workspace/custom_scripts/

# Copiar archivo de contenedor a Windows
docker cp pybullet_industrial_robotics:/workspace/PyBullet_Industrial_Robotics_Gym/Data/Model C:\pybullet-project\modelos

# Copiar carpeta completa
docker cp pybullet_industrial_robotics:/workspace/PyBullet_Industrial_Robotics_Gym/Data C:\pybullet-project\backup
```

### Limpieza y Mantenimiento

```powershell
# Ver espacio usado por Docker
docker system df

# Limpiar recursos no utilizados
docker system prune

# Limpiar todo (¡CUIDADO!)
docker system prune -a --volumes

# Reconstruir desde cero
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 🤖 Robots Disponibles

El proyecto soporta múltiples estructuras robóticas industriales:

| Robot | DOF | Nombre de Constante |
|-------|-----|---------------------|
| **Universal Robots UR3** | 6 | `Universal_Robots_UR3_Str` |
| **ABB IRB 120** | 6 | `ABB_IRB_120_Str` |
| **ABB IRB 120 + Eje Lineal** | 7 | `ABB_IRB_120_Str_7` |
| **Epson SCARA LS3-B401S** | 4 | `Epson_SCARA_LS3_B401S_Str` |
| **ABB IRB 14000 (YuMi)** | 14 (7+7) | `ABB_IRB_14000_Str` |

Para cambiar el robot, modificar en `train_td3.py`:
```python
CONST_ROBOT_TYPE = Parameters.<Nombre_de_Constante>
```

---

## 🎯 Environments Disponibles

### Environment E1 (Default)
- **Descripción:** Alcanzar objetivo estático o aleatorio sin obstáculos
- **Configuración:** `CONST_ENV_MODE = 'Default'`
- **Uso:** Entrenamiento base, aprendizaje de cinemática

### Environment E2 (Collision-Free)
- **Descripción:** Alcanzar objetivo con obstáculos de colisión estáticos
- **Configuración:** `CONST_ENV_MODE = 'Collision-Free'`
- **Uso:** Planificación con evasión de obstáculos

---

## 📈 Algoritmos DRL Implementados

### TD3 (Twin Delayed DDPG)
- **Archivo:** `train_td3.py`
- **Características:** Clipped double Q-learning, delayed policy updates
- **Recomendado para:** Espacios de acción continuos

### SAC (Soft Actor-Critic)
- **Archivo:** `train_sac.py`
- **Características:** Maximización de entropía, estabilidad en entrenamiento
- **Recomendado para:** Exploración robusta

### DDPG (Deep Deterministic Policy Gradient)
- **Archivo:** `train_ddpg.py`
- **Características:** Actor-critic determinístico
- **Recomendado para:** Baseline de comparación

### Extensión HER (Hindsight Experience Replay)
- **Configuración:** `CONST_ALGORITHM = 'TD3_HER'`
- **Características:** Aprendizaje de objetivos fallidos
- **Recomendado para:** Tareas de alcance con objetivos dispersos

---

## ⏱️ Tiempos de Entrenamiento Estimados

| Configuración | Timesteps | CPU (4 cores) | GPU |
|---------------|-----------|---------------|-----|
| **Prueba rápida** | 10,000 | 5-15 min | 2-5 min |
| **Entrenamiento corto** | 50,000 | 30-60 min | 10-20 min |
| **Entrenamiento completo** | 100,000 | 1-3 horas | 20-40 min |
| **Entrenamiento extenso** | 500,000 | 6-12 horas | 2-4 horas |

*Nota: Tiempos varían según CPU, robot seleccionado y complejidad del environment.*

---

## 🎓 Evaluación del Modelo

### Evaluación Básica

```bash
cd /workspace/PyBullet_Industrial_Robotics_Gym/Evaluation/Gym

# Evaluar environment
cd Environment
python test_env.py

# Evaluar modelo entrenado
cd ../Model
python test_model.py

# Control con modelo entrenado
cd ../Control
python test_model_control.py
```

### Exportar Modelo para Uso Externo

```powershell
# Copiar modelo entrenado a Windows
docker cp pybullet_industrial_robotics:/workspace/PyBullet_Industrial_Robotics_Gym/Data/Model/Environment_Default/TD3/Universal_Robots_UR3/model.zip C:\pybullet-project\modelo_final.zip
```

---

## 📚 Estructura del Proyecto

```
PyBullet_Industrial_Robotics_Gym/
├── URDFs/                          # Modelos URDF de robots
│   ├── UR3/
│   ├── IRB_120/
│   ├── SCARA/
│   └── ...
├── src/                            # Código fuente
│   ├── Industrial_Robotics_Gym/    # Gym environments
│   ├── PyBullet/                   # Wrappers de PyBullet
│   ├── RoLE/                       # Biblioteca de robótica
│   └── core.py                     # Core de simulación
├── Training/                       # Scripts de entrenamiento
│   ├── train_td3.py
│   ├── train_sac.py
│   └── train_ddpg.py
├── Evaluation/                     # Scripts de evaluación
│   ├── Gym/
│   └── PyBullet/
└── Data/                          # Datos generados (volumen)
    ├── Model/
    ├── Training/
    └── Prediction/
```

---

## 🔒 Consideraciones de Seguridad

- Los contenedores se ejecutan sin privilegios elevados
- Los datos persisten en volúmenes locales del host
- No se exponen puertos innecesarios
- Las credenciales no se almacenan en la imagen

---

## 🐛 Solución de Problemas Comunes

### Contenedor no inicia
```powershell
# Verificar logs
docker-compose logs

# Verificar recursos disponibles
docker system df
```

### Falta de espacio en disco
```powershell
# Limpiar recursos
docker system prune -a
```

### Rendimiento lento
- Aumentar CPUs y RAM en Docker Desktop Settings
- Reducir `batch_size` en los scripts de entrenamiento
- Reducir complejidad de red neuronal

---

## 📖 Referencias

- **Repositorio Original:** https://github.com/rparak/PyBullet_Industrial_Robotics_Gym
- **Paper de Investigación:** [Deep-Reinforcement-Learning-Based Motion Planning](https://www.mdpi.com/2079-3197/12/6/116)
- **PyBullet:** https://pybullet.org/
- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
- **Gymnasium:** https://gymnasium.farama.org/

---

## 👥 Información del Despliegue

- **Fecha de Despliegue:** Noviembre 2024
- **Plataforma:** Docker en Windows 10/11 con WSL 2
- **Environment:** E1 (Default - Sin Obstáculos)
- **Configuración:** CPU-only, modo headless (DIRECT)
- **Robots Testeados:** Universal Robots UR3

---

## 📝 Notas Adicionales

### Optimizaciones Realizadas

1. **Uso de imágenes slim** para reducir tamaño
2. **Instalación sin caché** de paquetes Python
3. **PyTorch CPU-only** para reducir tamaño de imagen
4. **Clonación shallow** del repositorio (--depth 1)
5. **Limpieza de archivos temporales** post-instalación

### Mejoras Futuras

- [ ] Soporte para GPU (NVIDIA Docker)
- [ ] Interfaz web para monitoreo en tiempo real
- [ ] Jupyter Notebook integrado
- [ ] CI/CD para entrenamiento automatizado
- [ ] Soporte para entrenamiento distribuido

---

## ✅ Checklist de Despliegue

- [x] Docker Desktop instalado y configurado
- [x] WSL 2 habilitado
- [x] Estructura de directorios creada
- [x] Dockerfile configurado
- [x] docker-compose.yml configurado
- [x] Imagen construida exitosamente
- [x] Contenedor iniciado
- [x] Scripts modificados (device='cpu')
- [x] Modo DIRECT configurado
- [x] Entrenamiento ejecutado
- [x] Resultados generados y accesibles

---

**Estado del Proyecto:** ✅ **Operacional**

**Última Actualización:** Noviembre 2024