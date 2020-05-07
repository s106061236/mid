/*------------------------------------Include-------------------------------------------*/
#include "DA7212.h"
DA7212 audio;
#include "mbed.h"
#include <cmath>
#include "uLCD_4DGL.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

/*-----------------------------------Define---------------------------------------------*/

#define DO 261
#define RE 294
#define MI 330
#define FA 349
#define SO 392
#define LAb 415
#define LA 440
#define SI 494
#define Do 523
#define Re 587
#define Mib 622
#define Mi 659
#define X  0

/*----------------------------------Global Declare---------------------------------------*/

int16_t waveform[kAudioTxBufferSize];
volatile int current_song = 0;
volatile int mode = 0;
volatile int current_cont = 0;
volatile int select_mode = 0;
uLCD_4DGL uLCD(D1, D0, D2);
InterruptIn btn_mode(SW2);
InterruptIn btn_cont(SW3);

/*---------------------------------Interrupt Function------------------------------------*/

void change_mode()
{
   mode = !mode;
}

void change_control()
{
   if(current_cont==2)
   {
      current_cont = 0;
   }
   else
   {
      current_cont = current_cont+1;
   }
   
}

/*-------------------------------------Song Array---------------------------------------*/

volatile int song_star[47] = {
  DO, DO, SO, SO, LA, LA, SO, SO, 
  FA, FA, MI, MI, RE, RE, DO, DO,
  SO, SO, FA, FA, MI, MI, RE, RE,
  SO, SO, FA, FA, MI, MI, RE, RE,
  DO, DO, SO, SO, LA, LA, SO, SO,
  FA, FA, MI, MI, RE, RE, DO};
/*
volatile int noteLength_star[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};*/

volatile int song_yamaha[47] = {
  DO, RE, MI, FA, SO, SO, 
  LA, FA, MI, MI, RE, RE, DO, DO, DO, DO,
  SO, FA, MI, SO, FA, MI, RE, RE,
  SO, FA, MI, SO, FA, MI, RE, RE,
  DO, RE, MI, FA, SO, SO,
  LA, FA, MI, MI, RE, RE, DO, DO,DO};
/*
volatile int noteLength_yamaha[47] ={
   1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2,
   1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2,
   1, 1, 1, 1, 1, 1, 2,
   1, 1, 1, 1, 1, 1, 2,
   1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1};*/

volatile int song_alice[47] ={
   Mi, Mib, Mi, Mib, Mi, SI, Re, Do, LA, LA, LA,
   DO, MI, LA, SI, SI, SI,
   MI, LAb, SI, Do, Do, Do,
   MI, Mi, Mib, Mi, Mib, Mi, SI, Re, Do, LA, LA, LA,
   DO, MI, LA, SI, SI, SI,
   MI, Do, SI, LA, LA, LA};
/*
volatile int noteLength_alice[41] ={
    1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
    1, 1, 1, 2, 1,
    1, 1, 1, 2, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
    1, 1, 1, 2, 1,
    1, 1, 1, 2, 1}; */

/*------------------------------------Audio Function------------------------------------*/

void playNote(int freq)
{
  for (int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
    audio.spk.play(waveform, kAudioTxBufferSize);

}

void playSong()
{
  int i,j;
  if(current_song==0)
  {
    //uLCD.printf("\nSong: Twinkle Star\n");
    for(i = 0; i < 47; i++)
    {
          for(j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
          {
            playNote(song_star[i]);
          }
          wait(0.2);
    }
  }
  else if(current_song==1)
  {
    //uLCD.printf("\nSong: YAMAHA\n");
    for(i = 0; i < 47; i++)
    {
          for(j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
          {
            playNote(song_yamaha[i]);
          }
          wait(0.2);
    }
  }
  else
  { 
    //uLCD.printf("\nSong: Alice\n");
    for(i = 0; i < 47; i++)
    {
          for(j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
          {
            playNote(song_alice[i]);
          }
          wait(0.2);
    }
  } 
}

/*----------------------------------DNN Predict Function-------------------------------*/

int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

/*-------------------------------DNN main Function--------------------------------------*/

int main(int argc, char* argv[]) {
  btn_mode.rise(&change_mode);
  btn_cont.rise(&change_control);
  uLCD.printf("Michael Chu 106061236\n");
  while(true){  
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");
  error_reporter->Report("%d\n",current_song);
  while (true) {
    
    if(mode==1)
    {
      mode=0;
       playSong();
    }
    else
    {
      if(select_mode ==1 )
      {
        current_song = current_cont;
        uLCD.locate(13,0);
        uLCD.printf("%d",current_song+1);
        
      }
    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);
    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      if(config.output_message[gesture_index]=="1")
      {
        select_mode = 0;
         if(current_song==0) current_song=2;
         else current_song = current_song-1;
         error_reporter->Report("%d\n",current_song);
         uLCD.cls();
         uLCD.printf("Now Song is: %d\n",current_song+1);
      }
      else if(config.output_message[gesture_index]=="2")
      {
        select_mode = 0;
         if(current_song==2) current_song=0;
         else current_song = current_song+1;
         error_reporter->Report("%d\n",current_song);
         uLCD.cls();
         uLCD.printf("Now Song is: %d\n",current_song+1);
      }
      else
      { 
         select_mode = 1;
         uLCD.cls();
         uLCD.printf("Select Mode: %d\n",current_cont+1);
      }   
    }
    }
  }
  }
}