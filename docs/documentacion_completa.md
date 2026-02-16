# Investigación sobre JAX

## ¿Qué es JAX?

JAX es una librería de Python desarrollada por **Google** para computación numérica de alto rendimiento y machine learning. Es usada actualmente en proyectos importantes como **AlphaFold** y **Gemini** de DeepMind.

### Características Principales

1. **API tipo NumPy**: Fácil de aprender si conoces NumPy
2. **Transformaciones composables**: grad, jit, vmap, pmap
3. **Alto rendimiento**: Compilación XLA optimizada
4. **Multi-plataforma**: CPU, GPU, TPU sin cambiar código
5. **Programación funcional**: Arrays inmutables y funciones puras

### Las 4 Transformaciones Clave

#### 1. **grad** - Diferenciación Automática
```python
from jax import grad
def f(x):
    return x**3 + 2*x**2

df = grad(f)  # Calcula la derivada automáticamente
```

#### 2. **jit** - Compilación Just-In-Time
```python
from jax import jit
@jit
def fast_function(x):
    return x @ x  # Se compila con XLA (2-10x más rápido)
```

#### 3. **vmap** - Vectorización Automática
```python
from jax import vmap
# Convierte función de un elemento a función de batch automáticamente
batch_fn = vmap(single_element_fn)
```

#### 4. **pmap** - Paralelización Multi-GPU
```python
from jax import pmap
# Distribuye cómputo en múltiples GPUs/TPUs
parallel_fn = pmap(fn)
```

---

## Comparación con TensorFlow y PyTorch

| Aspecto | JAX | PyTorch | TensorFlow |
|---------|-----|---------|------------|
| **Estilo** | Funcional | Orientado a objetos | Híbrido |
| **API** | NumPy | Propia | Keras/TF |
| **Abstracciones** | Bajo nivel | Medio nivel | Alto nivel |
| **Flexibilidad** | Máxima | Alta | Media |
| **Ecosistema** | Creciente | Maduro | Muy maduro |
| **Uso principal** | Investigación | Investigación/Producción | Producción |

### ¿Cuándo usar cada uno?

**JAX**: Investigación avanzada, máximo rendimiento, TPUs, nuevos algoritmos

**PyTorch**: Investigación general, prototipado rápido, ecosistema amplio

**TensorFlow**: Producción a gran escala, deployment móvil/edge

---

## Ecosistema JAX

### Frameworks de Redes Neuronales
- **Flax**: Framework oficial de Google, flexible y bien documentado
- **Haiku**: Simple y minimalista, por DeepMind
- **Equinox**: Moderno, con PyTrees ejecutables

### Librerías Especializadas
- **Optax**: Optimizadores (Adam, SGD, etc.)
- **NumPyro**: Programación probabilística
- **Chex**: Testing y debugging
- **Diffrax**: Ecuaciones diferenciales
- **JAX M.D.**: Dinámica molecular

### Proyectos Notables
- AlphaFold (predicción de proteínas)
- Gemini (LLM de Google)
- PaLM (Large Language Models)
- Imagen (generación de imágenes)

---

## Ventajas y Desventajas

### ✅ Ventajas
- Máximo rendimiento con XLA
- Transformaciones composables únicas
- API familiar (NumPy)
- Perfecto para TPUs
- Control total del código

### ❌ Desventajas
- Curva de aprendizaje (programación funcional)
- Ecosistema menos maduro que PyTorch
- Debugging más complejo con JIT
- Menos herramientas de deployment
- Requiere librerías externas para NN (no tiene nn.Module)

---

## Conclusión

JAX es una librería poderosa y moderna que destaca por su rendimiento y flexibilidad. Es ideal para:
- Investigación en machine learning
- Implementación de algoritmos nuevos
- Proyectos que requieren máximo rendimiento
- Trabajo con TPUs de Google Cloud

Su adopción por DeepMind y Google en proyectos de vanguardia demuestra su potencial. Aunque tiene una curva de aprendizaje, la inversión vale la pena para investigación avanzada.

---

## Referencias

- [Documentación oficial de JAX](https://docs.jax.dev/)
- [Repositorio GitHub](https://github.com/google/jax)
- [Awesome JAX](https://github.com/n2cholas/awesome-jax) - Recursos curados
- [Tutorial JAX-101](https://jax.readthedocs.io/en/latest/jax-101/index.html)
