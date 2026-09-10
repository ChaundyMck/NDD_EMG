import processing.serial.*;
import processing.opengl.*;

int num_cols = 10;
int bgcolor;                 // Background color
int fgcolor;                 // Fill color
Serial myPort;  // The serial port
PrintWriter MatOutput;
PrintWriter TargetOutput;

int[] serialInArray = new int[100];    // Where we'll put what we receive
int[] arrayWithTime = new int[101];
int[] pastInArray = new int [100];
float[][] colorTarget   = new float[3][100];
float[][] currentColor   = new float[3][100];
PVector[][] vertices = new PVector[num_cols][num_cols];
float[] verticesTZ = new float[num_cols];
float w = 30;
float ease = 0.75; 

int serialCount = 0;                 // A count of how many bytes we receive
int xpos, ypos;                  // Starting position of the ball
boolean firstContact = false;        // Whether we've heard from the microcontroller
int tiempoant;
boolean render=false;
int dif=0;

int comPortIdx = 0;

// Target Variables
int targetX, targetY;
int targetRadius = 40;
boolean targetVisible = false;
int appearTime = 0;
int nextDelay = 0;
int targetCount = 0;
int maxTargets = 120;

void settings() {
  // Fullscreen option did not work so exact coordinates for this laptop used
 // size(950, 800, P3D);    // or fullScreen(OPENGL); if you prefer
 fullScreen(P3D,1);
}

void setup() {  // Stage size
  noStroke();      // No border on the next thing draw
  
  surface.setLocation(0, 0);     // top-left corner
  surface.setResizable(false);   // optional
  surface.setTitle("EMG Task");
  
  // Print a list of the serial ports, for debugging purposes:
  println(Serial.list());
  
  String matfilename = "mat_data_" + nf(month(),2) + nf(day(),2) + nf(year(),4) + "_" + nf(hour(),2) + "-" + nf(minute(),2) + ".csv";
  String matsavefolder = "C:/Users/chaundymck/Indiana University/[Sec-E] BL-Phys-NDD_EMG - Documents/NDD_EMG/Data/Raw/mat_data/";
  String matfilepath = matsavefolder + matfilename;
  MatOutput = createWriter(matfilepath);  
  
  String targetfilename = "target_data_" + nf(month(),2) + nf(day(),2) + nf(year(),4) + "_" + nf(hour(),2) + "-" + nf(minute(),2) + ".csv";
  String targetsavefolder = "C:/Users/chaundymck/Indiana University/[Sec-E] BL-Phys-NDD_EMG - Documents/NDD_EMG/Data/Raw/target_data/";
  String targetfilepath = targetsavefolder + targetfilename;
  TargetOutput = createWriter(targetfilepath); 
  TargetOutput.println("targetX,targetY,appearTime,hitX,hitY,hitTime");
  
  myPort = new Serial(this, Serial.list()[comPortIdx], 115200);
  myPort.write('A');
  for (int j = 0; j < num_cols; j++) {
        for (int i = 0; i < num_cols; i++) {
            vertices[i][j] = new PVector( i*w, j*w, 0);
        }
    }
   

    
  println("Setup Done");
}

void draw() {
  background(255);
  if (targetCount < maxTargets) {
    if (!targetVisible && millis() >= nextDelay) {
      targetX = int(random(width/3 + targetRadius, 2*width/3 - targetRadius));
      targetY = int(random(height/3 + targetRadius, 2*height/3 - targetRadius));
      appearTime = millis();
      targetVisible = true;
    }
    
    if (targetVisible) {
      fill (255, 0, 0);
      ellipse(targetX, targetY, targetRadius*2, targetRadius*2);
    }
  } else {
    fill(0);
    text("Done!", width/2 -150, height/2);
  }
  //if (render==true) {
    
  //  translate(width/4, 100);
  //  rotateX(0.5);
  //  //rotateX(PI/10);
  //  background(0);
  //  for (int j=0; j<num_cols-1; j++) {
  //    beginShape(QUAD_STRIP);
  //    for (int i=0; i<num_cols; i++) {
  //        stroke(255);
     
  //        fill(serialInArray[j*num_cols+i], 0, 0);
  //        float x = i*width/num_cols;
  //        float y = j*height/num_cols;
  //        verticesTZ[i] = serialInArray[j*num_cols+i];
          
  //        vertices[i][j].z += (verticesTZ[i]-vertices[i][j].z)*ease;
  //        vertex( vertices[i][j].x, vertices[i][j].y, vertices[i][j].z);
  //        vertex( vertices[i][j+1].x, vertices[i][j+1].y, vertices[i][j+1].z);
  //      }
  //       endShape(CLOSE);
  //      //        println();
  //    }
  //    render=false;
  //}
}

void serialEvent(Serial myPort) {
  try{
  String data_string = myPort.readStringUntil('\n');
  arrayWithTime[0] = millis();
  
  if(data_string == null)
  {
    return;
  }
  data_string = data_string.trim();
  println("actual:" + data_string);
  if (data_string.startsWith("abc") == false)
  {
    println("leaving start");
    return;
  }
  if (data_string.endsWith("xyz") == false)
  {
    println("leaving end");
    return;
  }
  String [] split_data = split(data_string, ',');
  
  for (int i = 0; i < num_cols; i++)
  {
    for (int j = 0; j < num_cols; j++)
    {
      int idx = (i*num_cols) + j;
      serialInArray[idx] = int(split_data[idx+1]);
      arrayWithTime[idx +1] = int(split_data[idx+1]);
    }
  }
  if (targetCount > 1){
    MatOutput.println(join(str(arrayWithTime), ","));
    MatOutput.flush();
  }
    if (targetCount >= maxTargets){
    MatOutput.flush();
    MatOutput.close();
  }
    
    println(millis()-tiempoant);
     tiempoant = millis();
  render = true; 
    } catch (Exception e) {
    println("serialEvent error: " + e);
    e.printStackTrace();
  }
}

void mousePressed() {
  if (targetVisible && targetCount < maxTargets) {
    float d = dist(mouseX, mouseY, targetX, targetY);
    if (d < targetRadius) {
      int hitTime = millis();
      
      TargetOutput.println(targetX + "," + targetY + "," + appearTime + "," + mouseX + "," + mouseY + "," + hitTime);
      TargetOutput.flush();
      
      myPort.write("HIT\n"); //Tell arduino the target was hit
      
      targetVisible = false;
      targetCount++;
      scheduleNextTarget();
      
      if (targetCount >= maxTargets){
        TargetOutput.flush();
        TargetOutput.close();
      }
    }
  }
}

void scheduleNextTarget(){
   int wait = int(random(700,2500));
   nextDelay = millis() + wait;
}

void keyPressed() {
  MatOutput.flush();
  MatOutput.close();
  TargetOutput.flush();
  TargetOutput.close();
  exit();
}
  
