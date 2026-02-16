# Investigación sobre JAX

Actividad de investigación sobre **JAX**, la librería de Google para computación numérica de alto rendimiento y machine learning, utilizada en proyectos como **AlphaFold** y **Gemini** de DeepMind.

---

## 📁 Contenido del Repositorio

```
jax/
├── README.md                           # Este archivo
├── docs/documentacion_completa.md      # Documentación principal
├── ejemplos_jax.ipynb                  # Notebook con ejemplos prácticos
├── requirements.txt                    # Dependencias
└── ejercicio_jax.pdf                   # Enunciado del ejercicio
```

---

## 📚 Documentación

### [Ver Documentación Completa](docs/documentacion_completa.md)

La documentación cubre:

✅ **¿Qué es JAX?** - Características principales y transformaciones (grad, jit, vmap, pmap)

✅ **Comparación con TensorFlow y PyTorch** - Ventajas, desventajas y cuándo usar cada uno

✅ **Ecosistema JAX** - Flax, Haiku, Optax, NumPyro y proyectos notables

✅ **Casos de uso** - Investigación, rendimiento, TPUs

---

## 💻 Ejemplos Prácticos

### [Abrir Notebook de Ejemplos](ejemplos_jax.ipynb)

El notebook incluye:

1. **Transformaciones básicas de JAX**
   - Diferenciación automática (`grad`)
   - Compilación JIT (`jit`)
   - Vectorización (`vmap`)

2. **Regresión Lineal**
   - Implementación con gradient descent
   - Visualización de resultados

3. **Red Neuronal**
   - Clasificación del dataset Iris
   - Training loop completo
   - Métricas y gráficas

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd jax
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Abrir el notebook

```bash
jupyter notebook ejemplos_jax.ipynb
```

---

## 🎯 Puntos Tratados

Según el enunciado del ejercicio:

| Punto | Contenido |
|-------|-----------|
| ✅ **1. Qué es JAX** | Documentación completa + ejemplos |
| ✅ **2. Comparación** | Tabla comparativa con TF y PyTorch |
| ✅ **3. Ecosistema** | Librerías y herramientas principales |
| ✅ **4. Ejemplos prácticos** | Notebook interactivo con 3 ejemplos |

---

## 📦 Dependencias

```
jax[cpu]>=0.4.20
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
jupyter>=1.0.0
```

**Nota**: Instalación por defecto para CPU. Para GPU, consulta [documentación oficial](https://github.com/google/jax#installation).

---

## 📚 Referencias

- [Documentación oficial de JAX](https://docs.jax.dev/)
- [Repositorio GitHub](https://github.com/google/jax)
- [Tutorial JAX-101](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Awesome JAX](https://github.com/n2cholas/awesome-jax) - Recursos curados

---

## 👤 Autor

Investigación realizada para el **Máster de FP en Inteligencia Artificial y Big Data**

Fecha: Febrero 2026

---

**¡Gracias por visitar este repositorio!** ⭐
