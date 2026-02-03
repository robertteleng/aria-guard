# Docker Setup para ARIA Demo

## Por que Docker?

El Aria SDK (FastDDS) tiene incompatibilidades con ciertas versiones de glibc.
En particular, Ubuntu 24.04 con glibc 2.39-0ubuntu8.7 causa crashes al conectar
con las gafas Aria:

```
munmap_chunk(): invalid pointer
free(): invalid size
```

Docker permite usar Ubuntu 22.04 con una version compatible de glibc.

## Requisitos

- Docker con soporte GPU (nvidia-container-toolkit)
- Aria glasses conectadas por USB o WiFi

### Instalar nvidia-container-toolkit (si no lo tienes)

```bash
# Anadir repositorio
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Instalar
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configurar Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Uso Rapido

```bash
# Construir y ejecutar (primera vez)
docker compose up --build

# Ejecutar (si ya esta construido)
docker compose up

# Parar y limpiar
docker compose down
```

La aplicacion estara disponible en: http://localhost:5000

## Comandos Utiles

```bash
# Ver logs en tiempo real
docker compose logs -f

# Ejecutar shell dentro del contenedor
docker compose exec aria-demo bash

# Reconstruir sin cache
docker compose build --no-cache

# Ver estado de contenedores
docker compose ps
```

## Ejecucion Manual (sin compose)

```bash
# Construir imagen
docker build -t aria-demo .

# Ejecutar con GPU y USB
docker run -it --rm --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -v $(pwd)/data:/app/data \
  -p 5000:5000 \
  --gpus all \
  aria-demo
```

## Transferir a otro equipo (ej: Jetson)

```bash
# En el equipo origen: guardar imagen
docker save aria-demo:latest | gzip > aria-demo.tar.gz

# Copiar a destino
scp aria-demo.tar.gz usuario@jetson:/ruta/

# En el destino: cargar imagen
gunzip -c aria-demo.tar.gz | docker load

# Ejecutar
docker run -it --rm --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  -p 5000:5000 \
  --runtime nvidia \
  aria-demo
```

**Nota para Jetson**: La imagen x86 no funcionara directamente en ARM64.
Debes reconstruir la imagen en el Jetson o usar una imagen base ARM compatible.

## Troubleshooting

### Puerto 5000 ocupado

```bash
# Ver que usa el puerto
lsof -i :5000

# Parar contenedores anteriores
docker compose down
docker rm -f aria-demo
```

### No detecta las gafas Aria

```bash
# Verificar que las gafas estan conectadas
lsusb | grep -i aria

# Verificar que el contenedor tiene acceso USB
docker compose exec aria-demo lsusb
```

### Error de GPU

```bash
# Verificar que Docker ve la GPU
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
