<?php
header("Content-Type: application/json");

// Get the directory path from the form data
$directory_path = $_GET['directory_path'];

// Open the directory and get a list of files
$dir_handle = opendir($directory_path);
if (!$dir_handle) {
    die("Could not open directory: " . $directory_path);
}

$files = array();
while (($filename = readdir($dir_handle)) !== false) {
    // Filter out the "." and ".." directories
    if ($filename !== "." && $filename !== "..") {
        $filepath = $directory_path . '/' . $filename;
        $is_directory = is_dir($filepath);
        $files[] = array(
            'name' => $filename,
            'is_directory' => $is_directory
        );
    }
}
closedir($dir_handle);

// Convert the array of objects to JSON format and print it
$json_text = json_encode($files);
echo $json_text;
?>