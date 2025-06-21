#region Copyright
////////////////////////////////////////////////////////////////////////////////
// The following FIT Protocol software provided may be used with FIT protocol
// devices only and remains the copyrighted property of Garmin Canada Inc.
// The software is being provided on an "as-is" basis and as an accommodation,
// and therefore all warranties, representations, or guarantees of any kind
// (whether express, implied or statutory) including, without limitation,
// warranties of merchantability, non-infringement, or fitness for a particular
// purpose, are specifically disclaimed.
//
// Copyright 2015 Garmin Canada Inc.
////////////////////////////////////////////////////////////////////////////////
#endregion
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.IO;

using Dynastream.Fit;

namespace HrToRecordMesgBroadcastDemo
{
   class Program
   {
      static FileStream fitSource;

      static void Main(string[] args)
      {
         if (args.Length != 1)
         {
           //////console......WriteLine("Usage: decode.exe <filename>");
            return;
         }

         try
         {
            // Attempt to open .FIT file
            fitSource = new FileStream(args[0], FileMode.Open);
           //////console......WriteLine("Opening {0}", args[0]);

            //Attempt to create an output file
            String fileName = String.Format("{0}.csv",args[0].Split('.')); //Strip off the first part of the file name
            FileStream fs = new FileStream(fileName, FileMode.Create);
            // First, save the standard output.
            StreamWriter sw = new StreamWriter(fs);
            sw.AutoFlush = true;
           //////console......SetOut(sw);
         }
         catch(Exception ex)
         {
           //////console......WriteLine("DecodeDemo caught Exception: " + ex.Message);
            return;
         }

         Decode decodeDemo = new Decode();
         BufferedMesgBroadcaster mesgBroadcaster = new BufferedMesgBroadcaster();

         // Connect the Broadcaster to our event (message) source (in this case the Decoder)
         decodeDemo.MesgEvent += mesgBroadcaster.OnMesg;
         decodeDemo.MesgDefinitionEvent += mesgBroadcaster.OnMesgDefinition;

         // Subscribe to message events of interest by connecting to the Broadcaster
         mesgBroadcaster.MesgEvent += new MesgEventHandler(OnMesg);

         IMesgBroadcastPlugin plugin = new HrToRecordMesgBroadcastPlugin();
         mesgBroadcaster.RegisterMesgBroadcastPlugin(plugin);

         // Process the file

         try
         {
            //Attempt to decode the file
           //////console......WriteLine("Type,Local Number,Message,Field 1,Value 1,Units 1,Field 2,Value 2,Units 2,Field 3,Value 3,Units 3,Field 4,Value 4,Units 4,Field 5,Value 5,Units 5,Field 6,Value 6,Units 6");
            decodeDemo.Read(fitSource);
            mesgBroadcaster.Broadcast();
         }
         catch (FitException ex)
         {
           //////console......WriteLine("DecodeDemo caught FitException: " + ex.Message);
         }

         fitSource.Close();
         return;
      }

      #region Message Handlers

      static void OnMesg(object sender, MesgEventArgs e)
      {
         Mesg msg = e.mesg;

         if( msg.Num ==  MesgNum.Record)
         {
            RecordMesg recordMesg = new RecordMesg(msg);
           //////console......Write("Data,{0},record,", msg.LocalNum);
            if(recordMesg.GetTimestamp() != null)
            {
              //////console......Write("timestamp,{0},s,", recordMesg.GetTimestamp().GetTimeStamp());
            }
            if ( ( recordMesg.GetDistance() != null ) && ( recordMesg.GetDistance() != (uint)Fit.BaseType[Fit.UInt32].invalidValue ) )
            {
              //////console......Write("distance, {0:0.0}, m,", recordMesg.GetDistance());
            }
            if ( ( recordMesg.GetSpeed() != null ) && ( recordMesg.GetSpeed() != (ushort)Fit.BaseType[Fit.UInt16].invalidValue ) )
            {
              //////console......Write("speed,{0:0.000},m/s,", recordMesg.GetSpeed());
            }
            if ( ( recordMesg.GetCadence() != null ) && ( recordMesg.GetCadence() != (byte)Fit.BaseType[Fit.UInt8].invalidValue ) )
            {
              //////console......Write("cadence,{0},rpm,", recordMesg.GetCadence());
            }
            if ( ( recordMesg.GetEnhancedSpeed() != null ) && ( recordMesg.GetEnhancedSpeed() != (uint)Fit.BaseType[Fit.UInt32].invalidValue ) )
            {
              //////console......Write("enhanced_speed,{0:0.000},m/s,", recordMesg.GetEnhancedSpeed());
            }
            if ( ( recordMesg.GetHeartRate() != null ) && ( recordMesg.GetHeartRate() != (byte)Fit.BaseType[Fit.UInt8].invalidValue ) )
            {
              //////console......Write("heart_rate,{0},bpm,", recordMesg.GetHeartRate());
            }
           //////console......Write("\n");
         }
         else if( msg.Num == MesgNum.Hr )
         {
            HrMesg hrMesg = new HrMesg(msg);
           //////console......Write("Data,{0},hr,", msg.LocalNum);
            int count;
            if ( ( hrMesg.GetTimestamp() != null ) && ( hrMesg.GetTimestamp().GetTimeStamp() != (uint)Fit.BaseType[Fit.UInt32].invalidValue ) )
            {
               //////console......Write("timestamp,{0},,", hrMesg.GetTimestamp().GetTimeStamp());
            }
            if ( hrMesg.GetNumFilteredBpm() > 0 )
            {
               //////console......Write("filtered_bpm,");
                count = hrMesg.GetNumFilteredBpm();
                for (int i = 0; i < count; i++)
                {
                   //////console......Write("{0}", hrMesg.GetFilteredBpm(i));
                    if (i < count - 1)
                    {
                       //////console......Write("|");
                    }
                }
               //////console......Write(",bpm,");
            }
            if ( hrMesg.GetNumEventTimestamp12() > 0 )
            {
               //////console......Write("event_timestamp_12,");
                count = hrMesg.GetNumEventTimestamp12();
                for (int i = 0; i < count; i++)
                {
                   //////console......Write("{0}", hrMesg.GetEventTimestamp12(i));
                    if (i < count - 1)
                    {
                       //////console......Write("|");
                    }
                }
               //////console......Write(",,");
            }
            if ( hrMesg.GetNumEventTimestamp() > 0 )
            {
               //////console......Write("event_timestamp,");
                count = hrMesg.GetNumEventTimestamp();
                for (int i = 0; i < count; i++)
                {
                   //////console......Write("{0:G}", hrMesg.GetEventTimestamp(i));
                    if (i < count - 1)
                    {
                       //////console......Write("|");
                    }
                }
               //////console......Write(",s,");
            }
            if (hrMesg.GetFractionalTimestamp() != null)
            {
               //////console......Write("fractional_timestamp,{0:0.######},s,", hrMesg.GetFractionalTimestamp());
            }
           //////console......Write("\n");
         }
      }
      #endregion
   }
}
