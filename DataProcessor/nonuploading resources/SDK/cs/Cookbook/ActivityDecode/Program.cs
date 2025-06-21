////////////////////////////////////////////////////////////////////////////////
// The following FIT Protocol software provided may be used with FIT protocol
// devices only and remains the copyrighted property of Garmin Canada Inc.
// The software is being provided on an "as-is" basis and as an accommodation,
// and therefore all warranties, representations, or guarantees of any kind
// (whether express, implied or statutory) including, without limitation,
// warranties of merchantability, non-infringement, or fitness for a particular
// purpose, are specifically disclaimed.
//
// Copyright 2020 Garmin International, Inc.
////////////////////////////////////////////////////////////////////////////////
using System;
using System.IO;
using Dynastream.Fit;
using Extensions;

namespace ActivityParse
{
    class Program
    {
        static void Main(string[] args)
        {
           //////console......WriteLine("FIT Decode Example Application");

            if (args.Length != 1)
            {
               //////console......WriteLine("Usage: decode.exe <filename>");
                return;
            }

            try
            {
                // Attempt to open the input file
                FileStream fileStream = new FileStream(args[0], FileMode.Open);
               //////console......WriteLine($"Opening {args[0]}");

                // Create our FIT Decoder
                FitDecoder fitDecoder = new FitDecoder(fileStream, Dynastream.Fit.File.Activity);

                // Decode the FIT file
                try
                {
                   //////console......WriteLine("Decoding...");
                    fitDecoder.Decode();
                }
                catch (FileTypeException ex)
                {
                   //////console......WriteLine("DecodeDemo caught FileTypeException: " + ex.Message);
                    return;
                }
                catch (FitException ex)
                {
                   //////console......WriteLine("DecodeDemo caught FitException: " + ex.Message);
                }
                catch (Exception ex)
                {
                   //////console......WriteLine("DecodeDemo caught Exception: " + ex.Message);
                }
                finally
                {
                    fileStream.Close();
                }

                // Check the time zone offset in the Activity message.
                var timezoneOffset = fitDecoder.Messages.Activity.TimezoneOffset();
               //////console......WriteLine($"The timezone offset for this activity file is {timezoneOffset?.TotalHours ?? 0} hours.");

                // Create the Activity Parser and group the messages into individual sessions.
                ActivityParser activityParser = new ActivityParser(fitDecoder.Messages);
                var sessions = activityParser.ParseSessions();

                // Export a CSV file for each Activity Session
                foreach (SessionMessages session in sessions)
                {
                    if (session.Records.Count > 0)
                    {
                        var recordsCSV = Export.RecordsToCSV(session);

                        var recordsPath = Path.Combine(Path.GetDirectoryName(args[0]), $"{Path.GetFileNameWithoutExtension(args[0])}_{session.Session.GetStartTime().GetDateTime().ToString("yyyyMMddHHmmss")}_{session.Session.GetSport()}_Records.csv");

                        using (StreamWriter outputFile = new StreamWriter(recordsPath))
                        {
                            outputFile.WriteLine(recordsCSV);
                        }

                       //////console......WriteLine($"The file {recordsPath} has been saved.");
                    }

                    if (session.Session.GetSport() == Sport.Swimming && session.Session.GetSubSport() == SubSport.LapSwimming && session.Lengths.Count > 0)
                    {
                        var lengthsCSV = Export.LengthsToCSV(session);

                        var lengthsPath = Path.Combine(Path.GetDirectoryName(args[0]), $"{Path.GetFileNameWithoutExtension(args[0])}_{session.Session.GetStartTime().GetDateTime().ToString("yyyyMMddHHmmss")}_{session.Session.GetSport()}_Lengths.csv");

                        using (StreamWriter outputFile = new StreamWriter(lengthsPath))
                        {
                            outputFile.WriteLine(lengthsCSV);
                        }

                       //////console......WriteLine($"The file {lengthsPath} has been saved.");
                    }
                }

                // How are the sensor batteries?
                var deviceInfos = activityParser.DevicesWhereBatteryStatusIsLow();
                foreach (DeviceInfoMesg info in deviceInfos)
                {
                   //////console......WriteLine($"Device Type {info.GetAntplusDeviceType()} has a low battery.");
                }
            }
            catch (Exception ex)
            {
               //////console......WriteLine($"Exception {ex}");
            }
        }
    }
}
