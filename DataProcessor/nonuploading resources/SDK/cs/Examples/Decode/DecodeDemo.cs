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
// Copyright 2012 Garmin Canada Inc.
////////////////////////////////////////////////////////////////////////////////
#endregion

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Diagnostics;
using Dynastream.Fit;


namespace DecodeDemo
{
    class Program
    {
        static Dictionary<ushort, int> mesgCounts = new Dictionary<ushort, int>();
        static FileStream fitSource;

        static void Main(string[] args)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            //////////console.WriteLine("FIT Decode Example Application");

            if (args.Length != 1)
            {
                //////////console.WriteLine("Usage: decode.exe <filename>");
                return;
            }

            try
            {
                // Attempt to open .FIT file
                fitSource = new FileStream(args[0], FileMode.Open);
                //////////console.WriteLine("Opening {0}", args[0]);

                Decode decodeDemo = new Decode();
                MesgBroadcaster mesgBroadcaster = new MesgBroadcaster();

                // Connect the Broadcaster to our event (message) source (in this case the Decoder)
                decodeDemo.MesgEvent += mesgBroadcaster.OnMesg;
                decodeDemo.MesgDefinitionEvent += mesgBroadcaster.OnMesgDefinition;
                decodeDemo.DeveloperFieldDescriptionEvent += OnDeveloperFieldDescriptionEvent;

                // Subscribe to message events of interest by connecting to the Broadcaster
                mesgBroadcaster.MesgEvent += OnMesg;
                mesgBroadcaster.MesgDefinitionEvent += OnMesgDefn;

                mesgBroadcaster.FileIdMesgEvent += OnFileIDMesg;
                mesgBroadcaster.UserProfileMesgEvent += OnUserProfileMesg;
                mesgBroadcaster.MonitoringMesgEvent += OnMonitoringMessage;
                mesgBroadcaster.DeviceInfoMesgEvent += OnDeviceInfoMessage;
                mesgBroadcaster.RecordMesgEvent += OnRecordMessage;

                bool status = decodeDemo.IsFIT(fitSource);
                status &= decodeDemo.CheckIntegrity(fitSource);

                // Process the file
                if (status)
                {
                    //////////console.WriteLine("Decoding...");
                    decodeDemo.Read(fitSource);
                    //////////console.WriteLine("Decoded FIT file {0}", args[0]);
                }
                else
                {
                    try
                    {
                        //////////console.WriteLine("Integrity Check Failed {0}", args[0]);
                        if (decodeDemo.InvalidDataSize)
                        {
                            //////////console.WriteLine("Invalid Size Detected, Attempting to decode...");
                            decodeDemo.Read(fitSource);
                        }
                        else
                        {
                            //////////console.WriteLine("Attempting to decode by skipping the header...");
                            decodeDemo.Read(fitSource, DecodeMode.InvalidHeader);
                        }
                    }
                    catch (FitException ex)
                    {
                        //////////console.WriteLine("DecodeDemo caught FitException: " + ex.Message);
                    }
                }
                fitSource.Close();

                //////////console.WriteLine("");
                //////////console.WriteLine("Summary:");
                int totalMesgs = 0;
                foreach (KeyValuePair<ushort, int> pair in mesgCounts)
                {
                    //////////console.WriteLine("MesgID {0,3} Count {1}", pair.Key, pair.Value);
                    totalMesgs += pair.Value;
                }

                //////////console.WriteLine("{0} Message Types {1} Total Messages", mesgCounts.Count, totalMesgs);

                stopwatch.Stop();
                //////////console.WriteLine("");
                //////////console.WriteLine("Time elapsed: {0:0.#}s", stopwatch.Elapsed.TotalSeconds);
                //////////console.ReadKey();
            }
            catch (FitException ex)
            {
                //////////console.WriteLine("A FitException occurred when trying to decode the FIT file. Message: " + ex.Message);
            }
            catch (Exception ex)
            {
                //////////console.WriteLine("Exception occurred when trying to decode the FIT file. Message: " + ex.Message);
            }
        }

        private static void OnDeveloperFieldDescriptionEvent(object sender, DeveloperFieldDescriptionEventArgs args)
        {
            //////////console.WriteLine("New Developer Field Description");
            //////////console.WriteLine("   App Id: {0}", args.Description.ApplicationId);
            //////////console.WriteLine("   App Version: {0}", args.Description.ApplicationVersion);
            //////////console.WriteLine("   Field Number: {0}", args.Description.FieldDefinitionNumber);
        }

        #region Message Handlers
        // Client implements their handlers of interest and subscribes to MesgBroadcaster events
        static void OnMesgDefn(object sender, MesgDefinitionEventArgs e)
        {
            //////////console.WriteLine("OnMesgDef: Received Defn for local message #{0}, global num {1}", e.mesgDef.LocalMesgNum, e.mesgDef.GlobalMesgNum);
            //////////console.WriteLine("\tIt has {0} fields {1} developer fields and is {2} bytes long",
                e.mesgDef.NumFields,
                e.mesgDef.NumDevFields,
                e.mesgDef.GetMesgSize());
        }

        static void OnMesg(object sender, MesgEventArgs e)
        {
            //////////console.WriteLine("OnMesg: Received Mesg with global ID#{0}, its name is {1}", e.mesg.Num, e.mesg.Name);

            int i = 0;
            foreach (Field field in e.mesg.Fields)
            {
                for (int j = 0; j < field.GetNumValues(); j++)
                {
                    //////////console.WriteLine("\tField{0} Index{1} (\"{2}\" Field#{4}) Value: {3} (raw value {5})",
                        i,
                        j,
                        field.GetName(),
                        field.GetValue(j),
                        field.Num,
                        field.GetRawValue(j));
                }

                i++;
            }

            foreach (var devField in e.mesg.DeveloperFields)
            {
                for (int j = 0; j < devField.GetNumValues(); j++)
                {
                    //////////console.WriteLine("\tDeveloper{0} Field#{1} Index{2} (\"{3}\") Value: {4} (raw value {5})",
                        devField.DeveloperDataIndex,
                        devField.Num,
                        j,
                        devField.Name,
                        devField.GetValue(j),
                        devField.GetRawValue(j));
                }
            }

            if (mesgCounts.ContainsKey(e.mesg.Num))
            {
                mesgCounts[e.mesg.Num]++;
            }
            else
            {
                mesgCounts.Add(e.mesg.Num, 1);
            }
        }

        static void OnFileIDMesg(object sender, MesgEventArgs e)
        {
            //////////console.WriteLine("FileIdHandler: Received {1} Mesg with global ID#{0}", e.mesg.Num, e.mesg.Name);
            FileIdMesg myFileId = (FileIdMesg)e.mesg;
            try
            {
                //////////console.WriteLine("\tType: {0}", myFileId.GetType());
                //////////console.WriteLine("\tManufacturer: {0}", myFileId.GetManufacturer());
                //////////console.WriteLine("\tProduct: {0}", myFileId.GetProduct());
                //////////console.WriteLine("\tSerialNumber {0}", myFileId.GetSerialNumber());
                //////////console.WriteLine("\tNumber {0}", myFileId.GetNumber());
                //////////console.WriteLine("\tTimeCreated {0}", myFileId.GetTimeCreated());

                //Make sure properties with sub properties arent null before trying to create objects based on them
                if (myFileId.GetTimeCreated() != null)
                {
                    Dynastream.Fit.DateTime dtTime = new Dynastream.Fit.DateTime(myFileId.GetTimeCreated().GetTimeStamp());
                }
            }
            catch (FitException exception)
            {
                //////////console.WriteLine("\tOnFileIDMesg Error {0}", exception.Message);
                //////////console.WriteLine("\t{0}", exception.InnerException);
            }
        }

        static void OnUserProfileMesg(object sender, MesgEventArgs e)
        {
            //////////console.WriteLine("UserProfileHandler: Received {1} Mesg, it has global ID#{0}", e.mesg.Num, e.mesg.Name);
            UserProfileMesg myUserProfile = (UserProfileMesg)e.mesg;
            string friendlyName;
            try
            {
                try
                {
                    friendlyName = myUserProfile.GetFriendlyNameAsString();
                }
                catch (ArgumentNullException)
                {
                    //There is no FriendlyName property
                    friendlyName = "";
                }
                //////////console.WriteLine("\tFriendlyName \"{0}\"", friendlyName);
                //////////console.WriteLine("\tGender {0}", myUserProfile.GetGender().ToString());
                //////////console.WriteLine("\tAge {0}", myUserProfile.GetAge());
                //////////console.WriteLine("\tWeight  {0}", myUserProfile.GetWeight());
            }
            catch (FitException exception)
            {
                //////////console.WriteLine("\tOnUserProfileMesg Error {0}", exception.Message);
                //////////console.WriteLine("\t{0}", exception.InnerException);
            }
        }

        static void OnDeviceInfoMessage(object sender, MesgEventArgs e)
        {
            //////////console.WriteLine("DeviceInfoHandler: Received {1} Mesg, it has global ID#{0}", e.mesg.Num, e.mesg.Name);
            DeviceInfoMesg myDeviceInfoMessage = (DeviceInfoMesg)e.mesg;
            try
            {
                //////////console.WriteLine("\tTimestamp  {0}", myDeviceInfoMessage.GetTimestamp());
                //////////console.WriteLine("\tBattery Status{0}", myDeviceInfoMessage.GetBatteryStatus());
            }
            catch (FitException exception)
            {
                //////////console.WriteLine("\tOnDeviceInfoMesg Error {0}", exception.Message);
                //////////console.WriteLine("\t{0}", exception.InnerException);
            }
        }

        static void OnMonitoringMessage(object sender, MesgEventArgs e)
        {
            //////////console.WriteLine("MonitoringHandler: Received {1} Mesg, it has global ID#{0}", e.mesg.Num, e.mesg.Name);
            MonitoringMesg myMonitoringMessage = (MonitoringMesg)e.mesg;
            try
            {
                //////////console.WriteLine("\tTimestamp  {0}", myMonitoringMessage.GetTimestamp());
                //////////console.WriteLine("\tActivityType {0}", myMonitoringMessage.GetActivityType());
                switch (myMonitoringMessage.GetActivityType()) // Cycles is a dynamic field
                {
                    case ActivityType.Walking:
                    case ActivityType.Running:
                        //////////console.WriteLine("\tSteps {0}", myMonitoringMessage.GetSteps());
                        break;
                    case ActivityType.Cycling:
                    case ActivityType.Swimming:
                        //////////console.WriteLine("\tStrokes {0}", myMonitoringMessage.GetStrokes());
                        break;
                    default:
                        //////////console.WriteLine("\tCycles {0}", myMonitoringMessage.GetCycles());
                        break;
                }
            }
            catch (FitException exception)
            {
                //////////console.WriteLine("\tOnDeviceInfoMesg Error {0}", exception.Message);
                //////////console.WriteLine("\t{0}", exception.InnerException);
            }
        }

        private static void OnRecordMessage(object sender, MesgEventArgs e)
        {
            //////////console.WriteLine("Record Handler: Received {0} Mesg, it has global ID#{1}",
                e.mesg.Num,
                e.mesg.Name);

            var recordMessage = (RecordMesg)e.mesg;

            WriteFieldWithOverrides(recordMessage, RecordMesg.FieldDefNum.HeartRate);
            WriteFieldWithOverrides(recordMessage, RecordMesg.FieldDefNum.Cadence);
            WriteFieldWithOverrides(recordMessage, RecordMesg.FieldDefNum.Speed);
            WriteFieldWithOverrides(recordMessage, RecordMesg.FieldDefNum.Distance);

            WriteDeveloperFields(recordMessage);
        }

        private static void WriteDeveloperFields(Mesg mesg)
        {
            foreach (var devField in mesg.DeveloperFields)
            {
                if (devField.GetNumValues() <= 0)
                {
                    continue;
                }

                if (devField.IsDefined)
                {
                    //////////console.Write("\t{0}", devField.Name);

                    if (devField.Units != null)
                    {
                        //////////console.Write(" [{0}]", devField.Units);
                    }
                    //////////console.Write(": ");
                }
                else
                {
                    //////////console.Write("\tUndefined Field: ");
                }

                //////////console.Write("{0}", devField.GetValue(0));
                for (int i = 1; i < devField.GetNumValues(); i++)
                {
                    //////////console.Write(",{0}", devField.GetValue(i));
                }

                //////////console.WriteLine();
            }
        }

        private static void WriteFieldWithOverrides(Mesg mesg, byte fieldNumber)
        {
            Field profileField = Profile.GetField(mesg.Num, fieldNumber);
            bool nameWritten = false;

            if (null == profileField)
            {
                return;
            }

            IEnumerable<FieldBase> fields = mesg.GetOverrideField(fieldNumber);

            foreach (FieldBase field in fields)
            {
                if (!nameWritten)
                {
                    //////////console.WriteLine("   {0}", profileField.GetName());
                    nameWritten = true;
                }

                if (field is Field)
                {
                    //////////console.WriteLine("      native: {0}", field.GetValue());
                }
                else
                {
                    //////////console.WriteLine("      override: {0}", field.GetValue());
                }
            }
        }

        #endregion
    }
}
