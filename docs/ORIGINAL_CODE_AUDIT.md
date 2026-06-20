# Auditoria do Código Original

Este documento audita os scripts e notebooks exploratórios originais
(`C:\Users\rafael\PycharmProjects\pythonProject3_TCC`) que deram origem às
figuras e à narrativa publicadas. O objetivo é separar o que é **confiável e
reproduzível** do que contém **bugs ou inconsistências**, para guiar o que foi
portado para `src/vibration_id/`.

Regra geral adotada no port: **portar = curar, não copiar.** Os originais são
código de pesquisa (caminhos absolutos, imports duplicados, parâmetros
contraditórios entre versões). Só o que está descrito abaixo como "confiável"
foi migrado.

## Mapa: figura publicada → código original

| Figura no repositório | Código original | Estado |
| --- | --- | --- |
| `figures/advanced/havok_reconstruction.png` | `recon.py` (Hankel + SVD + SINDy linear) | confiável |
| `figures/advanced/sindy_identification.png` | `sindy_tratado.py` | confiável |
| `figures/advanced/duffing_global_envelope_fit.png` | `prof/definitivo.ipynb` (otimização global, narrativa §11) | **portado** (`global_fit.py`) |
| `figures/sensor/sensor_nonlinearity_fit.png` | `sss.ipynb` (output map `h(q)`) | **confiável — melhor formulação** |
| `figures/sensor/sindy_linear_model_subtraction.png` | `completo.py` (resíduo do modelo linear) | **portado** (`sensor_residual.py`, bug corrigido) |
| `figures/sensor/sensor_residual_*` | `completo.py` (análise de resíduo) | **portado** (`sensor_residual.py`) |
| classificação de material | `euler bernoulli.py` | já portado e corrigido |

## O que é confiável (e foi portado)

### 1. SINDy restrito + mapa de saída do sensor — `sss.ipynb`

É a formulação mais coerente de todo o material. Separa explicitamente:

- **Dinâmica** (Duffing livre): `v' = -ω₀² q - γ v - α q³`, identificada por STLSQ
  sobre a biblioteca `[q, v, q³]`.
- **Mapa de saída do sensor** (não-linearidade estática): `h(q) = a₀ + a₁q + a₂q² + a₃q³`,
  ajustado por mínimos quadrados entre o estado reconstruído e o sinal medido.

Essa separação dinâmica-vs-sensor é exatamente a ideia central da narrativa
(seção 4). Portado para `src/vibration_id/nonlinear_id.py`.

### 1b. Ajuste global Duffing com amortecimento não-linear — `prof/definitivo.ipynb`

Procedimento em 3 estágios (narrativa §11), portado para `global_fit.py`:

1. **Envoltória não-linear** (curve_fit) sobre o envelope de Hilbert, modelo
   `r(t) = sqrt( e^(-γt) / (1/r0² + (η/4γ)(1 - e^(-γt))) )` → recupera `γ` (linear)
   e `η` (não-linear).
2. **Calibração de fase/frequência** (curve_fit em `A e^(-γt) cos(ωt+φ)` no início)
   → `ω` preciso e `φ` (condições iniciais). Crucial: ajustar MSE direto num sinal
   longo é multimodal em frequência; este estágio trava `ω` antes do refino.
3. **Otimização global** (Nelder-Mead) de `(ω₀², β, φ)` com `γ, η` fixos,
   minimizando o MSE da simulação de `x'' = -ω₀²x - βx³ - (γ+ηx²)x'`.

Reproduz a narrativa: `γ≈0.353, η≈0.078, ω²≈13840 (f≈18.72 Hz)`. O `β` é
fracamente identificável (na narrativa variou 172/−755/−6.9 entre runs), logo a
não-linearidade dominante é na dissipação, não na rigidez.

### 2. SVD de ensemble — `Untitled.ipynb`

Interpola N experimentos num eixo de tempo comum, monta a matriz `X (M × N)` e
aplica SVD; reconstrução de posto baixo (`r=1,2`) com erro de Frobenius
relativo. Tecnicamente correto. Portado para `src/vibration_id/ssa.py`
(`ensemble_svd` / `reconstruct_rank`).

### 3. HAVOK — `recon.py`

Hankel (`stack_size=1000`) + SVD + SINDy linear sobre os modos. Já coberto por
`src/vibration_id/havok.py`. A única observação é o uso de `M_eff_g = 9.335 g`
como massa efetiva **incluindo o alvo de papel** — confirma a necessidade da
correção de massa de ponta no modelo Euler-Bernoulli (ver abaixo).

### 4. Fator de qualidade por meia-potência — `fatorQ.py`

Usa o método correto `Q = f_r / Δf`, com `Δf` medido na banda de meia-potência
(−3 dB) ao redor do pico. Portado para `src/vibration_id/damping.py`
(`quality_factor_half_power`). O repositório limpo vinha usando apenas a
convenção `f₀/γ`, que **não** é o Q físico (falta o fator 2π).

## O que tem bug ou inconsistência (NÃO portar como está)

### A. `completo.py` — `DT = 1000` usado como passo de derivada

Na linha que define `DT = 1000` e depois chama
`savgol_filter(x, ..., deriv=1, delta=DT)`, o `delta` deveria ser o passo de
amostragem em segundos (`1e-3`), não `1000`. Isso escala a velocidade por um
fator `1e6`, invalidando qualquer parâmetro físico derivado dessa velocidade.
**Corrigido no port** (`sensor_residual.py` / `run_sensor_residual.py`): a
velocidade usa `velocity_savgol(t, q, ...)`, que deriva `delta` do vetor de
tempo real.

### B. `completo.py` — três modelos SINDy contraditórios

O mesmo arquivo define `omega0_sq` como `981.94`, `983.308` e `1060.851` em
funções diferentes, além de coeficientes não-lineares inconsistentes
(`+39.8 x²`, `-52 x²`, `+3 x³`). São tentativas diferentes deixadas lado a lado;
nenhuma é "a" identificação final. Use a formulação de `sss.ipynb`.

### C. `euler bernoulli.py` — geometria divergente do comentário

Na função teórica, `L = 0.27` e `h = 0.003` enquanto os comentários dizem
"30 cm" e "2.33 mm". A função inversa usa `L = 0.30`, `h = 0.00233`. O port
(`materials.py`) adota explicitamente `L = 0.30 m`, `h = 2.33 mm` para o plástico
e `L = 0.27 m`, `h = 1.50 mm` para o inox, documentados em
`data/sample/material_trials.csv`.

### D. Euler-Bernoulli sem massa de ponta

O modelo engastada-livre não considerava massa na ponta. Foi adicionada a
correção de massa efetiva (Rayleigh, `m_eff = 0.2427 rho A L + m_ponta`) em
`materials.py`, disponível via `--tip-mass-kg` (CLI) e estimada das dimensões
`tip_*` no estudo de materiais.

**Importante (resultado honesto):** para o caso inox, a massa do alvo de papel
(~0,21 g, de 35×25×0,3 mm a ~800 kg/m³) é **desprezível** frente à massa efetiva
da viga (~15 g), então a correção quase não altera o E (17,59 → 17,88 GPa). Ou
seja, a massa de ponta **não** explica o módulo ~10× baixo do inox. Um aço de
`L=0.27 m, h=1.5 mm` ressoaria perto de ~17 Hz no modelo ideal, mas o medido é
~5 Hz; isso aponta para **compliance no engaste / condição de contorno
não-ideal** (a régua não está perfeitamente bi-engastada), não para massa de
ponta. Por isso o README reporta o inox como "rigidez efetiva do conjunto", e
essa leitura continua correta. A correção de massa de ponta é, ainda assim, um
recurso válido para montagens em que a massa adicionada é relevante.

### E. `fatorQ.py` / `euler bernoulli.py` — `input()` e caminhos absolutos

Misturam I/O interativo, `plt.show()` e caminhos fixos de dataset. No port, toda
entrada vira argumento de função/CLI e nenhuma figura é aberta interativamente.

## Notebooks: papel de cada um

| Notebook | Conteúdo | Aproveitado |
| --- | --- | --- |
| `final.ipynb` | sincronização, corte por derivada, FFT individual e de ensemble, classificação de ruído 1/f | conceitos já no pipeline; ensemble-FFT é candidato a port futuro |
| `Untitled.ipynb` | SVD de ensemble entre experimentos | portado (`ssa.py`) |
| `sss.ipynb` | Duffing + mapa de saída do sensor | portado (`nonlinear_id.py`) |
