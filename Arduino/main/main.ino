const int BUZZER = 3;

const int CHIRP_START = 6000; // chirp start in Hz
const int CHIRP_FINISH = 8000; // chirp finish in Hz


const int buttonPin = 2;
const int ledPin =  LED_BUILTIN;

int buttonState = 0;         // variable for reading the pushbutton status
bool reset = true;

void setup(){
  pinMode(BUZZER, OUTPUT);
  pinMode(buttonPin, INPUT);

}

void chirp() {
  for (int i = CHIRP_START; i < CHIRP_FINISH; i++) {
    tone(BUZZER, i);
    delay(0.01);
  }
  noTone(BUZZER);
}

void loop() {
  buttonState = digitalRead(buttonPin);

  if (buttonState == HIGH && reset) {
    chirp();
    reset = false;
  } else if (buttonState == LOW) {
    reset = true;
  }

  delay(100);
}
