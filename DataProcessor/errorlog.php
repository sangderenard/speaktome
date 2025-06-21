<?php

// Read the contents of the PHP error log
$error_log = ini_get('error_log');
$log_contents = file_get_contents($error_log);

// Get the current timestamp and calculate the timestamp 8 hours ago
$current_time = time();
$eight_hours_ago = $current_time - (8 * 60 * 60);

// Split the log contents into an array of individual log entries
$log_entries = preg_split('/\r?\n/', $log_contents, -1, PREG_SPLIT_NO_EMPTY);

// Define the $matches variable before using it in preg_match
$matches = array();

// Reverse the order of the log entries array
$log_entries = array_reverse($log_entries);

// Loop through each log entry and only display entries with a timestamp within the last 8 hours
foreach ($log_entries as $log_entry) {
    preg_match('/^\[([^]]+)\]/', $log_entry, $matches);
    if (!empty($matches)) {
        $timestamp = strtotime($matches[1]);
        if ($timestamp >= $eight_hours_ago) {
            echo $log_entry . "<br>" . PHP_EOL;
        }
        
    }
}

?>