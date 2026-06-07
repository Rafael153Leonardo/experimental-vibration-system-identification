const int sensorPin = A0;
const unsigned long Ts_us = 1000; // 1000 Hz

unsigned long t0;
unsigned long nextSampleTime;

void setup() {
  // Maximum serial speed used for high-rate acquisition on Arduino Uno.
  // If the USB converter fails, reduce this to 500000.
  Serial.begin(2000000);

  pinMode(sensorPin, INPUT);

  // ADC hardware acceleration.
  // Arduino's default ADC prescaler is 128, which gives 125 kHz ADC clock.
  // Prescaler 32 gives 500 kHz ADC clock and reduces analogRead latency
  // from roughly 110 us to roughly 30 us with acceptable precision.
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
