<?php


// Generate some test data
// Generate some test data
$data = "This is some test data";

// Encrypt the test data using the server's public key
$public_key = openssl_pkey_get_public(file_get_contents(__DIR__ . "/public_key.pem"));
if (!$public_key) {
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "Failed to load public key.";
    exit();
}

$encrypted = "";
if (!openssl_public_encrypt($data, $encrypted, $public_key, OPENSSL_PKCS1_OAEP_PADDING)) {
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "Encryption failed.";
    exit();
}
//echo bin2hex($encrypted) . "\n\n";
// Decrypt the encrypted data using the server's private key
$private_key = openssl_pkey_get_private("file://" . __DIR__ . "/private_key.pem");

if (!$private_key) {
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "Failed to load private key.";
    exit();
}

$decrypted = "";
if (!openssl_private_decrypt($encrypted, $decrypted, $private_key, OPENSSL_PKCS1_OAEP_PADDING)) {
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "Decryption failed.";
    exit();
}

// Check if the decrypted data matches the original data
if ($decrypted != $data) {
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "Encryption and decryption failed.";
    exit();
}





// Get the uploaded file and payload
$file = $_FILES["file"];








$payload = $_POST["payload"];

echo "The payload is: " . $payload . "\n\n";;
$binaryPayload = base64_decode($payload);
$stringPayload = implode(array_map("chr",unpack('C*',$binaryPayload)));
echo "Interpreted as hex string: " . bin2hex($binaryPayload) . "\n\n";
echo "As a byte array string: " . $stringPayload . "\n\n";
openssl_private_decrypt($binaryPayload, $decryptedPayload, $private_key, OPENSSL_PKCS1_OAEP_PADDING);
echo "After decryption is looks like: " . $decryptedPayload . "\n\n";
$decryptedPayload = hex2bin($decryptedPayload);
echo "Interpreting that as a hex string, this is the binary: " . $decryptedPayload . "\n\n";

// Convert the decrypted payload into a string
$decrypted_string = "";
for ($i = 0; $i < strlen($decryptedPayload); $i++) {
    $decrypted_string .= chr(ord($decryptedPayload[$i]) & 0xFF);
}
echo $decrypted_string;

exit();










// Decode the JSON string into an array of encoded encrypted chunks
$encodedChunks = json_decode($payload, true);

// Create an empty array to hold the decrypted chunks
$decryptedChunks = array();

// Decrypt each chunk and store the decrypted data in $decryptedChunks
foreach ($encodedChunks as $encodedChunk) {
    $decodedChunk = base64_decode($encodedChunk);
    echo base64_encode($decodedChunk) . "\n\n";
    openssl_private_decrypt($decodedChunk, $decryptedChunk, $private_key, OPENSSL_PKCS1_OAEP_PADDING);
    $decryptedChunks[] = $decryptedChunk;
    echo $decryptedChunk;
}

// Concatenate the decrypted chunks to obtain the combined data
http_response_code(400);
header('Content-Type: text/plain');
echo count($decryptedChunks) . "\n\n" . $decryptedChunks[0] . "\n\n";
$combinedDataDecrypted = implode($decryptedChunks);

// Base64 decode the combined data to obtain the original data
$combinedData = base64_decode($combinedDataDecrypted);
// Extract the IV (initialization vector), encrypted password, encrypted client public key, and payload from the decoded data
// Extract the IV (initialization vector) from the first 16 bytes of the decoded data
$salt = hex2bin(hex2bin(substr(bin2hex($combinedData), 0, 64)));
$iv = hex2bin(hex2bin(substr(bin2hex($combinedData), 64, 64)));
$iterations = 10000; // set the number of iterations

// Generate the key using PBKDF2 and the hashed password
// Hash the valid password using SHA-256
$hashed_valid_password = hash('sha256', "mysecretkey");
http_response_code(400);
header('Content-Type: text/plain');
echo $hashed_valid_password . "\n" . bin2hex($salt) . "\n"; 
$key = hex2bin(hash_pbkdf2('sha256', $hashed_valid_password, $salt, $iterations, 64, false));
echo $key . "\n\n";
//$key = $hashed_valid_password;
// Check if the IV is 16 bytes long
error_log($iv);
if (strlen($iv) !== 16) {
    // IV is not 16 bytes long, log an error and return 400 response
    error_log("Invalid IV: " . $iv);
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "DataLength" . strlen($payload);
    echo "Invalid IV\n";
    echo "Decrypted IV: " . $iv;
    echo "Salt: " . $salt;
    exit();
}
$encrypted_password = hex2bin(hex2bin(substr(bin2hex($combinedData), 128, 320)));
$decrypted_password = openssl_decrypt($encrypted_password, 'aes-256-cbc', $key, OPENSSL_PKCS1_PADDING, $iv);
$hashed_password = $decrypted_password;


// Decrypt the password using the server's private key and AES-256-CBC
//$hashed_password = openssl_decrypt($encrypted_password, 'aes-256-cbc', $key, OPENSSL_ZERO_PADDING, $iv);



if ($hashed_password !== $hashed_valid_password) {
    // Password is invalid, return error response
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "Invalid password\n";
    echo "\nIV: " . bin2hex($iv);
    echo "\nSalt: " . bin2hex($salt);
    echo "\nKey: " . $key;
    echo "\nEncrypted Password: " . bin2hex($encrypted_password);
    echo "\nHashed Password: " . $hashed_password;
    echo "\nHashed Valid Password: " . $hashed_valid_password;
    exit();
}

$encrypted_client_public_key = hex2bin(hex2bin(substr(bin2hex($combinedData), 448, 1024)));
$payload_string = hex2bin(hex2bin(substr(bin2hex($combinedData), 1472)));


// Decrypt the client public key using the server's private key and AES-256-CBC
$decrypted_client_public_key = openssl_decrypt($encrypted_client_public_key, 'aes-256-cbc', $key, OPENSSL_PKCS1_PADDING, $iv);

// Import the client public key as an RSA key object
$client_public_key = openssl_pkey_get_public($decrypted_client_public_key);

// Decrypt the payload using the client's public key and AES-256-CBC
$decrypted_payload_string = openssl_decrypt($payload_string, 'aes-256-cbc', $key, OPENSSL_PKCS1_PADDING, $iv);

// Generate the upload path from the payload string
$id_string = $decrypted_payload;
$upload_path = "/home4/oktqajmy/public_html/bike-dogbite-me/cgi-bin/{$id_string}";

// If the upload directory does not exist, create it
if (!is_dir($upload_path)) {
    mkdir($upload_path, 0777, true);
}

// If the file data is defined, save the file to the server
if ($file['error'] === UPLOAD_ERR_OK) {
    $filename = $file['name'];
    $upload_path = "{$upload_path}/{$filename}";
    
    move_uploaded_file($file['tmp_name'], $upload_path);
    
    // Print a success message
    header('Content-Type: text/plain');
    echo "File uploaded successfully to {$upload_path}\n";
} else {
    // Print an error message if no file data was received
    http_response_code(400);
    header('Content-Type: text/plain');
    echo "No file data received\n";
}
?>