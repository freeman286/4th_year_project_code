const int BUZZER = 3; //buzzer to arduino pin 9


void setup(){
 
  pinMode(BUZZER, OUTPUT); // Set buzzer - pin 9 as an output

}

void loop(){
 
  tone(BUZZER, 1000); // Send 1KHz sound signal...
  delay(1000);        // ...for 1 sec
  noTone(BUZZER);     // Stop sound...
  delay(1000);        // ...for 1sec
  
}
