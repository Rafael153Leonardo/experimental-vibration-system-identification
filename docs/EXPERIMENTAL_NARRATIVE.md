# Narrativa Tecnica: Aquisicao Experimental, Calibracao do Sensor e Identificacao do Sistema

Este projeto nasceu de um arranjo experimental caseiro para medir vibracoes
mecanicas e transformar o sinal adquirido em parametros fisicos interpretaveis.
Construi a bancada, adaptei o sensor optico, adquiri os dados com Arduino e
desenvolvi o pipeline computacional em Python para filtragem, identificacao de
sistemas e ajuste de modelo fisico.

O objetivo principal foi partir de um sinal bruto real e chegar a uma descricao
quantitativa do sistema: frequencia natural, amortecimento, resposta nao linear
do sensor e modelo dinamico aproximado do oscilador.

## 1. Arranjo Experimental de Aquisicao

A bancada foi montada com uma estrutura mecanica simples, uma lamina/regua
vibrante e um alvo optico fixado na ponta do sistema. O deslocamento do alvo foi
medido lateralmente por um sensor refletivo TCRT5000 modificado, conectado a um
Arduino Uno.

![Full acquisition rig](../figures/setup/acquisition_rig_full.jpeg)

![Arduino and sensor circuit](../figures/setup/arduino_sensor_circuit.jpeg)

![Sensor and moving target alignment](../figures/setup/sensor_target_alignment.jpeg)

Usei o Arduino apenas como sistema de aquisicao. Ele nao aplicava calibracao,
filtragem ou conversao para deslocamento. A placa lia o valor analogico bruto do
sensor, registrava o tempo em microssegundos e enviava os dados pela serial no
formato CSV. Toda a interpretacao do sinal foi feita depois em Python.

Essa escolha deixou a etapa embarcada simples e deterministica. O Arduino ficou
responsavel por amostrar o sensor de forma regular, enquanto o computador ficou
responsavel por tratar ruido, sincronizar sinais, estimar frequencias e ajustar
modelos.

## 2. Codigo de Aquisicao no Arduino

O codigo usado no Arduino esta salvo em `hardware/arduino/AMM.ino`. A rotina
usa a entrada analogica `A0`, define um periodo de amostragem de `1000 us`
aproximadamente equivalente a `1000 Hz`, acelera a leitura do ADC e envia duas
colunas: `tempo_us` e `leitura_bruta`.

```cpp
const int sensorPin = A0;
const unsigned long Ts_us = 1000; // 1000 Hz

unsigned long t0;
unsigned long nextSampleTime;

void setup() {
  Serial.begin(2000000);
  pinMode(sensorPin, INPUT);

  ADCSRA &= ~(bit(ADPS0) | bit(ADPS1) | bit(ADPS2));
  ADCSRA |= bit(ADPS2) | bit(ADPS0);

  t0 = micros();
  nextSampleTime = micros();

  Serial.println("tempo_us,leitura_bruta");
}

void loop() {
  unsigned long now = micros();

  if ((long)(now - nextSampleTime) >= 0) {
    int raw_value = analogRead(sensorPin);

    Serial.print(now - t0);
    Serial.print(",");
    Serial.println(raw_value);

    nextSampleTime += Ts_us;
  }
}
```

As decisoes principais dessa etapa foram:

- usar o canal analogico `A0` para capturar a resposta continua do sensor;
- trabalhar com `Ts_us = 1000`, mantendo a amostragem em torno de `1000 Hz`;
- usar `Serial.begin(2000000)` para reduzir gargalo de transmissao;
- alterar o prescaler do ADC para diminuir a latencia de `analogRead`;
- salvar o dado bruto para preservar a informacao original do experimento.

## 3. Modificacao e Calibracao do Sensor TCRT5000

O TCRT5000 e um sensor optico refletivo composto por um emissor infravermelho e
um fototransistor. No uso comum, ele costuma atuar como detector de proximidade
ou de contraste. Neste projeto eu o usei de outra forma: como transdutor analogico
para acompanhar o deslocamento de um alvo vibrante.

Para isso, modifiquei o uso original do sensor:

- posicionei o par optico em relacao ao alvo movel;
- ajustei mecanicamente distancia e alinhamento;
- adaptei o circuito de condicionamento para leitura analogica;
- usei o Arduino para registrar a tensao bruta;
- recalculei a curva efetiva de resposta em Python usando os dados reais.

A curva do datasheet representa uma condicao idealizada: corrente relativa do
fototransistor em funcao da distancia, usando superficie e circuito de teste
padronizados. A curva que obtive no projeto nao e a curva isolada do componente.
Ela representa a resposta efetiva da cadeia completa de medicao:

```text
TCRT5000 + circuito modificado + alvo real + alinhamento mecanico + ADC do Arduino
```

Essa diferenca e importante porque o experimento real inclui geometria,
refletividade do alvo, montagem, ruido eletrico e limitacoes da aquisicao.

## 4. Reconstrucao Computacional da Resposta do Sensor

Depois de adquirir o sinal de vibracao, ajustei primeiro um modelo linear para a
dinamica mecanica dominante. Esse modelo descreve a parte principal do
oscilador: posicao, velocidade, rigidez linear e amortecimento linear.

Em seguida, subtrai o comportamento linear identificado do sinal medido. A parte
restante nao foi tratada apenas como ruido. Analisei esse residuo no tempo e no
espaco de fase para separar a dinamica mecanica principal da nao linearidade
introduzida pela cadeia de medicao.

![SINDy linear model subtraction](../figures/sensor/sindy_linear_model_subtraction.png)

Com essa estrategia, usei o residuo do modelo linear para recuperar uma correcao
nao linear associada ao sensor. A curva experimental `h(x)` representa como a
resposta medida se desvia do modelo linear esperado.

![Sensor nonlinearity fit](../figures/sensor/sensor_nonlinearity_fit.png)

Tambem analisei a assinatura do residuo por diferentes perspectivas:

![Sensor residual signature](../figures/sensor/sensor_residual_signature.png)

![Sensor residual spectrum](../figures/sensor/sensor_residual_spectrum.png)

![Sensor residual phase space](../figures/sensor/sensor_residual_phase_space.png)

![Sensor residual projection](../figures/sensor/sensor_residual_projection.png)

A comparacao com o datasheet do TCRT5000 foi feita de forma estrutural e
qualitativa. O datasheet fornece a curva optica padrao do componente; a minha
reconstrucao fornece a curva efetiva do sensor dentro da bancada real.

| Datasheet TCRT5000 | Curva reconstruida no experimento |
| --- | --- |
| Componente optico em condicao padrao de teste | Cadeia completa de medicao montada na bancada |
| Corrente relativa do fototransistor versus distancia | Correcao residual `h(x)` versus estado modelado |
| Superficie refletiva padronizada | Alvo real usado no sistema vibrante |
| Geometria controlada pelo fabricante | Distancia e alinhamento definidos pela montagem |
| Resposta optica do componente | Resposta efetiva: optica, circuito, mecanica e ADC |

## 5. Pipeline de Identificacao

Com a aquisicao e a resposta do sensor caracterizadas, organizei o pipeline para
transformar os CSVs brutos em resultados fisicos. O fluxo principal foi:

```text
dados brutos
  -> deteccao do inicio da vibracao
  -> correcao de tempo e offset
  -> inspecao visual dos sinais
  -> filtragem e reconstrucao por SVD
  -> estimativa de frequencia por FFT
  -> identificacao por SINDy
  -> ajuste de envoltoria por Hilbert
  -> simulacao por EDO
  -> otimizacao global dos parametros
```

O notebook `definitivo.ipynb` concentrou a etapa final dessa analise. Nele, os
sinais sincronizados foram limpos, comparados e usados para ajustar modelos
dinamicos progressivamente mais estruturados.

## 6. Preparacao dos Dados

No inicio da analise, carreguei os CSVs sincronizados e detectei o inicio da
oscilacao pelo maior salto de derivada. A partir desse ponto, cada sinal foi
deslocado para iniciar em `t = 0`.

Depois apliquei correcoes basicas:

- zeragem do eixo de tempo;
- remocao do offset do sinal;
- salvamento dos arquivos corrigidos;
- comparacao visual entre repeticoes experimentais.

Essa etapa foi importante para garantir que a identificacao nao fosse feita
sobre sinais desalinhados ou com deslocamento DC artificial.

## 7. Reconstrucao por SVD

Para reduzir ruido sem destruir a estrutura dinamica, usei uma abordagem baseada
em matriz de trajetoria e decomposicao SVD. A serie temporal foi reorganizada em
janelas, decomposta em valores singulares e reconstruida com os componentes mais
relevantes.

Na execucao principal, usei:

```text
janela L = 100
componentes mantidos = [0, 1]
```

A reconstrucao gerou a coluna `x_svd_limpo`, usada como entrada mais limpa para
as etapas de identificacao.

## 8. Identificacao por SINDy e Modelo Duffing

Com o sinal suavizado, preparei os estados de posicao e velocidade para aplicar
SINDy. O objetivo foi descobrir uma equacao candidata para o oscilador a partir
dos dados, sem impor manualmente todos os termos desde o inicio.

Um dos modelos identificados apresentou estrutura compativel com um oscilador
tipo Duffing:

```text
(x)' = 1.000 v
(v)' = -20.285 -13892.495 x -0.508 v + 1.132 x^2 -0.214 x v + 4.082 x^3
```

Tambem comparei configuracoes diferentes de SINDy:

```text
Model A R2 = 0.99973
Model B R2 = 0.99974
```

A interpretacao fisica foi que a rigidez linear domina a resposta, enquanto os
termos nao lineares aparecem de forma mais relevante na dissipacao de energia e
na correcao residual do sensor.

## 9. Derivadas Suavizadas e Amortecimento Nao Linear

Como a velocidade estimada por derivada numerica e sensivel a ruido, usei
suavizacao Savitzky-Golay antes de calcular derivadas. Isso melhorou o retrato
no espaco de fase e permitiu avaliar termos de amortecimento dependentes do
estado.

Os termos estimados nessa etapa foram:

```text
v term:     -0.05324
x v term:    0.12095
x^2 v term: -0.23613
```

Essa parte da analise mostrou que a perda de energia do sistema nao precisava
ser descrita apenas por amortecimento linear.

## 10. Ajuste da Envoltoria por Hilbert

Usei a transformada de Hilbert para extrair a envoltoria do sinal amortecido e
ajustar uma lei de decaimento. O ajuste retornou:

```text
gamma = 0.35865
eta   = 0.07722
R0    = 4.6580
```

Nesse modelo, `gamma` representa o amortecimento linear e `eta` representa uma
componente de amortecimento nao linear. Isso transformou a curva de decaimento
em parametros fisicos interpretaveis.

## 11. Simulacao por EDO e Otimizacao Global

Na etapa final, escrevi o oscilador como um sistema de EDOs de primeira ordem:

```text
x' = v
v' = -omega_n^2 x - beta x^3 - (gamma + eta x^2) v
```

Depois integrei numericamente esse modelo e ajustei os parametros para minimizar
o erro entre a simulacao e o sinal experimental. Tambem refinei fase e frequencia
para alinhar a trajetoria simulada com os dados reais.

A calibracao fina indicou:

```text
phase phi      = -0.0484 rad
omega          = 117.6848 rad/s
omega^2        = 13849.72
```

O ajuste global retornou:

```text
x0      = 4.2833 mm
v0      = -6.8996 mm/s
gamma   = 0.3629
eta     = 0.0809
omega^2 = 13907.74
beta    = -6.91
```

A figura organizada abaixo resume esse resultado, comparando o sinal ruidoso, o
sinal suavizado por SVD, o modelo numerico otimizado e a envoltoria analitica:

![Duffing global envelope fit](../figures/advanced/duffing_global_envelope_fit.png)

## 12. Interpretacao Final

O resultado principal do projeto foi a construcao de uma cadeia completa de
identificacao experimental:

- montei a bancada de aquisicao;
- adaptei o TCRT5000 para uso analogico;
- registrei dados reais com Arduino;
- tratei ruido, offset e sincronizacao em Python;
- usei FFT para estimar frequencia dominante;
- usei SVD para reconstruir sinais mais limpos;
- usei SINDy para investigar a estrutura dinamica;
- usei Hilbert para estimar amortecimento;
- usei simulacao por EDO para ajustar um modelo fisico global.

A conclusao tecnica e que o sistema medido se comporta principalmente como um
oscilador de rigidez linear, mas a dissipacao de energia e a cadeia de medicao
apresentam efeitos nao lineares relevantes. A reconstrucao residual permitiu
separar parte da nao linearidade do sensor da dinamica mecanica principal,
tornando o sensor um elemento modelado do experimento, e nao apenas uma fonte de
dado bruto.

## 13. Fontes Originais Usadas

Os arquivos originais usados para organizar esta narrativa foram:

```text
C:\Users\rafael\Documents\arduino\AMM\AMM.ino
C:\Users\rafael\Downloads\TCRT5000.PDF
C:\Users\rafael\PycharmProjects\pythonProject3_TCC\prof\definitivo.ipynb
```

O datasheet do TCRT5000 foi usado como referencia tecnica e nao e redistribuido
neste repositorio.
