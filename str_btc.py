import hashlib
import json
import urllib.request
import ecdsa
import base58

# secp256k1 curve order
SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140

def derive_private_key_from_string(string):
    # Hash the string to get a fixed-size input
    hashed_string = hashlib.sha256(string.encode()).digest()

    # Use the hashed string as a private key
    private_key = int.from_bytes(hashed_string, 'big')

    # Ensure the private key is within the valid range [1, n-1]
    private_key = (private_key % (SECP256K1_ORDER - 1)) + 1

    return private_key

def private_key_to_wif(private_key):
    # Convert the private key to a WIF-encoded string
    private_key_bytes = private_key.to_bytes(32, 'big')
    wif = base58.b58encode_check(b'\x80' + private_key_bytes).decode()

    return wif

def private_key_to_public_key(private_key):
    # Derive the public key from the private key using secp256k1
    private_key_bytes = private_key.to_bytes(32, 'big')
    signing_key = ecdsa.SigningKey.from_string(private_key_bytes, curve=ecdsa.SECP256k1)
    verifying_key = signing_key.get_verifying_key()

    # Compressed public key
    point = verifying_key.pubkey.point
    prefix = b'\x02' if point.y() % 2 == 0 else b'\x03'
    compressed = prefix + point.x().to_bytes(32, 'big')

    return compressed

def public_key_to_address(public_key_bytes):
    # SHA-256 then RIPEMD-160
    sha256_hash = hashlib.sha256(public_key_bytes).digest()
    ripemd160 = hashlib.new('ripemd160', sha256_hash).digest()

    # Base58Check encode with version byte 0x00 (mainnet)
    address = base58.b58encode_check(b'\x00' + ripemd160).decode()

    return address

def check_address_on_chain(address):
    """Check address activity via Blockstream's public API."""
    url = f"https://blockstream.info/api/address/{address}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "str_btc/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        stats = data.get("chain_stats", {})
        tx_count = stats.get("tx_count", 0)
        funded = stats.get("funded_txo_sum", 0)
        spent = stats.get("spent_txo_sum", 0)
        balance = funded - spent
        return {
            "tx_count": tx_count,
            "total_received": funded,
            "total_sent": spent,
            "balance": balance,
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <string>")
        sys.exit(1)

    string = sys.argv[1]
    private_key = derive_private_key_from_string(string)
    wif_private_key = private_key_to_wif(private_key)
    public_key = private_key_to_public_key(private_key)
    address = public_key_to_address(public_key)

    print("Private key (WIF):", wif_private_key)
    print("Public key (hex): ", public_key.hex())
    print("Address:          ", address)

    print("\nChecking address on blockchain...")
    info = check_address_on_chain(address)
    if "error" in info:
        print("  Could not check:", info["error"])
    elif info["tx_count"] == 0:
        print("  No transactions found — address has never been used.")
    else:
        sat_to_btc = lambda s: f"{s / 1e8:.8f} BTC"
        print(f"  Transactions:    {info['tx_count']}")
        print(f"  Total received:  {sat_to_btc(info['total_received'])}")
        print(f"  Total sent:      {sat_to_btc(info['total_sent'])}")
        print(f"  Balance:         {sat_to_btc(info['balance'])}")
        print(f"  Explorer:        https://blockstream.info/address/{address}")