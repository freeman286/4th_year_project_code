const int BUZZER = 3; //buzzer to arduino pin 9

const int CHIRP_START = 8000; // chirp start in Hz
const int CHIRP_FINISH = 12000; // chirp finish in Hz


void setup(){
 
  pinMode(BUZZER, OUTPUT); // Set buzzer - pin 9 as an output

}

void loop(){

  chirp();
  delay(1000);
  
}

void chirp() {
  for (int i = CHIRP_START; i < CHIRP_FINISH; i++) {
    tone(BUZZER, i);
    delay(0.01);
  }
  noTone(BUZZER);
}
