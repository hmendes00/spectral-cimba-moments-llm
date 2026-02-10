"""
Satoshi's Treasure Hunt - Candidate passphrase checker.

This script derives Bitcoin addresses from notable strings found in
Satoshi Nakamoto's public communications and checks each against the
blockchain via Blockstream's API.

Categories of candidates:
  1. Genesis block coinbase message (exact & variations)
  2. Whitepaper title & abstract phrases
  3. Key philosophical statements from forum posts
  4. First & last known communications
  5. Significant dates/numbers as strings
  6. Known Satoshi-associated metadata (email, name, birthday)
  7. Cryptographic/numerical easter eggs
  8. Famous quotes from posts
"""

import hashlib
import json
import time
import urllib.request
import ecdsa
import base58

# ── key derivation (same as str_btc.py) ──────────────────────────────

SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140


def derive_private_key(string):
    h = hashlib.sha256(string.encode()).digest()
    pk = int.from_bytes(h, "big")
    pk = (pk % (SECP256K1_ORDER - 1)) + 1
    return pk


def private_key_to_public(pk):
    sk = ecdsa.SigningKey.from_string(pk.to_bytes(32, "big"), curve=ecdsa.SECP256k1)
    vk = sk.get_verifying_key()
    pt = vk.pubkey.point
    prefix = b"\x02" if pt.y() % 2 == 0 else b"\x03"
    return prefix + pt.x().to_bytes(32, "big")


def public_key_to_address(pub):
    h = hashlib.sha256(pub).digest()
    r = hashlib.new("ripemd160", h).digest()
    return base58.b58encode_check(b"\x00" + r).decode()


def check_address(address):
    url = f"https://blockstream.info/api/address/{address}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "satoshi-hunt/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        stats = data.get("chain_stats", {})
        tx = stats.get("tx_count", 0)
        funded = stats.get("funded_txo_sum", 0)
        spent = stats.get("spent_txo_sum", 0)
        return tx, funded, spent
    except Exception as e:
        return None, None, str(e)


# ── CANDIDATE STRINGS ────────────────────────────────────────────────

CANDIDATES = [
    # ━━━ CATEGORY 1: Genesis Block ━━━
    # The most iconic string in all of Bitcoin
    ("Genesis coinbase (exact)", "The Times 03/Jan/2009 Chancellor on brink of second bailout for banks"),
    ("Genesis coinbase (no 'The')", "Times 03/Jan/2009 Chancellor on brink of second bailout for banks"),
    ("Genesis coinbase (lowercase)", "the times 03/jan/2009 chancellor on brink of second bailout for banks"),
    ("Genesis block hash", "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"),
    ("Genesis timestamp", "1231006505"),
    ("Genesis nonce", "2083236893"),
    ("Genesis date", "2009-01-03"),
    ("Genesis date slash", "03/01/2009"),
    ("Genesis date compact", "20090103"),
    ("Genesis reward address", "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"),

    # ━━━ CATEGORY 2: Whitepaper ━━━
    ("Whitepaper title", "Bitcoin: A Peer-to-Peer Electronic Cash System"),
    ("Whitepaper title (lower)", "bitcoin: a peer-to-peer electronic cash system"),
    ("Whitepaper abstract opening", "A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution."),
    ("Whitepaper abstract short", "A purely peer-to-peer version of electronic cash"),
    ("Whitepaper first sentence", "Commerce on the Internet has come to rely almost exclusively on financial institutions serving as trusted third parties to process electronic payments."),
    ("Whitepaper conclusion", "We have proposed a system for electronic transactions without relying on trust."),

    # ━━━ CATEGORY 3: Satoshi's Identity / Profile ━━━
    ("Satoshi Nakamoto", "Satoshi Nakamoto"),
    ("satoshi nakamoto", "satoshi nakamoto"),
    ("satoshinakamoto", "satoshinakamoto"),
    ("Satoshi", "Satoshi"),
    ("satoshi", "satoshi"),
    ("Nakamoto", "Nakamoto"),
    ("Email address", "satoshin@gmx.com"),
    ("satoshi@vistomail.com", "satoshi@vistomail.com"),
    ("P2P Foundation birthday", "1975-04-05"),
    ("P2P Foundation birthday v2", "April 5, 1975"),
    ("P2P Foundation birthday v3", "05/04/1975"),

    # ━━━ CATEGORY 4: Key Forum Post Quotes ━━━
    # First P2P Foundation post - opening line
    ("P2P Foundation opening", "I've developed a new open source P2P e-cash system called Bitcoin."),
    # Core philosophical statement
    ("Trust problem", "The root problem with conventional currency is all the trust that's required to make it work."),
    # The central bank statement
    ("Central bank trust", "The central bank must be trusted not to debase the currency, but the history of fiat currencies is full of breaches of that trust."),
    # Privacy/encryption analogy
    ("Encryption analogy", "Then strong encryption became available to the masses, and trust was no longer required."),
    ("Time for money", "It's time we had the same thing for money."),
    # Wikileaks post (last significant public statement)
    ("Wikileaks warning", "I make this appeal to WikiLeaks not to try to use Bitcoin."),
    ("Wikileaks heat", "the heat you would bring would likely destroy us at this stage"),
    # Bitcoin as digital gold analogy
    ("No single point", "The result is a distributed system with no single point of failure."),
    # Users hold keys
    ("Crypto keys", "Users hold the crypto keys to their own money and transact directly with each other"),

    # ━━━ CATEGORY 5: First Email to Cryptography List ━━━
    ("First email opening", "I've been working on a new electronic cash system that's fully peer-to-peer, with no trusted third party."),
    ("Bitcoin P2P e-cash paper", "Bitcoin P2P e-cash paper"),

    # ━━━ CATEGORY 6: Key Dates as Strings ━━━
    ("Whitepaper date", "2008-10-31"),
    ("Whitepaper date v2", "October 31, 2008"),
    ("Whitepaper date v3", "31/10/2008"),
    ("Whitepaper date v4", "20081031"),
    ("Bitcoin v0.1 release", "2009-01-09"),
    ("Bitcoin v0.1 release v2", "January 9, 2009"),
    ("First transaction date", "2009-01-12"),
    ("Last post date", "2010-12-12"),
    ("Last post date v2", "December 12, 2010"),
    ("Hal Finney tx date", "2009-01-12"),

    # ━━━ CATEGORY 7: Numerical / Cryptographic ━━━
    ("21 million", "21000000"),
    ("21 million with text", "21 million"),
    ("50 BTC", "50"),
    ("Block reward", "5000000000"),
    ("Max supply hex", "0x1312D00"),
    ("Bitcoin port", "8333"),
    ("Halving interval", "210000"),
    ("10 minutes", "600"),
    ("Difficulty 1", "1"),
    ("Satoshi unit", "100000000"),

    # ━━━ CATEGORY 8: Other Notable Phrases ━━━
    ("Chancellor bailout", "Chancellor on brink of second bailout for banks"),
    ("Second bailout", "second bailout for banks"),
    ("Crypto proof", "everything is based on crypto proof instead of trust"),
    ("P2P network", "peer-to-peer network"),
    ("Proof of work", "proof-of-work"),
    ("Double spending", "double-spending"),
    ("Block chain", "block chain"),
    ("Hash cash", "Hashcash"),
    ("Digital signatures", "digital signatures"),
    
    # ━━━ CATEGORY 9: Hidden References / Easter Eggs ━━━
    # Reversed genesis message
    ("Genesis reversed", "sknab rof tuoliab dnoces fo knirb no rollecnahC 9002/naJ/30 semiT ehT"),
    # Hex of genesis coinbase text
    ("Genesis coinbase hex", "5468652054696D65732030332F4A616E2F32303039204368616E63656C6C6F72206F6E206272696E6B206F66207365636F6E64206261696C6F757420666F722062616E6B73"),
    # Bitcoin.org
    ("bitcoin.org", "http://www.bitcoin.org"),
    ("bitcoin.org domain", "bitcoin.org"),
    # Software version strings
    ("Version 0.1", "Bitcoin v0.1"),
    ("Version 0.3.19", "Bitcoin 0.3.19"),
    
    # ━━━ CATEGORY 10: Potential Wordplay / Anagrams ━━━
    ("Name anagram: So a man took his", "So a man took his"),
    ("Name meaning", "central intelligence"),
    ("Name meaning 2", "born from the ashes"),
    ("Name meaning 3", "wisdom of the ages"),
    ("Name meaning 4", "aah, it took masons"),
    # Combination of key concepts
    ("Trust no one", "trust no one"),
    ("In code we trust", "In code we trust"),
    ("In crypto we trust", "In crypto we trust"),
    ("In math we trust", "In math we trust"),
    ("Vires in numeris", "Vires in numeris"),
    ("Chancellor", "Chancellor"),
    
    # ━━━ CATEGORY 11: Possible concatenations ━━━
    ("Name + genesis date", "Satoshi Nakamoto 2009-01-03"),
    ("Name + genesis msg", "Satoshi Nakamoto The Times 03/Jan/2009 Chancellor on brink of second bailout for banks"),
    ("Whitepaper URL", "http://www.bitcoin.org/bitcoin.pdf"),
    ("bitcoin.pdf", "bitcoin.pdf"),
    
    # ━━━ CATEGORY 12: Ascii art from source code ━━━
    # Satoshi's sourceforge acct
    ("Sourceforge", "s_nakamoto"),
    # Known test strings from early Bitcoin code
    ("Hello world hash", "Hello, world!"),
    ("Test", "test"),
    ("Bitcoin", "Bitcoin"),
    ("bitcoin", "bitcoin"),

    # ━━━ CATEGORY 13: Double-Space Habit (typewriter era) ━━━
    # Satoshi used double spaces after periods — try key phrases with that quirk
    ("Genesis dbl-space", "The Times 03/Jan/2009 Chancellor on brink of second bailout for banks.  The beginning."),
    ("Trust dbl-space", "The root problem with conventional currency is all the trust that's required to make it work.  With e-currency based on cryptographic proof, without the need to trust a third party middleman, money can be secure and transactions effortless."),
    ("Moved on dbl-space", "I've moved on to other things.  It's in good hands with Gavin and everyone."),
    ("Farewell dbl-space", "I've moved on to other things."),
    ("Farewell exact", "I've moved on to other things"),
    ("Farewell to Gavin", "I've moved on to other things. It's in good hands with Gavin and everyone."),

    # ━━━ CATEGORY 14: British English Clues ━━━
    # Satoshi used British spellings despite "Japanese" name
    ("British: colour", "colour"),
    ("British: favourite", "favourite"),
    ("British: honour", "honour"),
    ("British: grey", "grey"),
    ("British: analyse", "analyse"),
    ("British: defence", "defence"),
    ("British: bloody", "bloody hell"),
    ("British: mate", "Cheers, mate"),
    ("British: Chancellor", "Chancellor of the Exchequer"),
    ("British Times ref", "The Times"),
    ("British newspaper", "The Times of London"),
    # Satoshi specifically chose a British newspaper — maybe combined with identity
    ("Name + Times", "Satoshi Nakamoto - The Times"),
    ("Times + date", "The Times 03/Jan/2009"),

    # ━━━ CATEGORY 15: "I've moved on" Farewell (2011) ━━━
    # Satoshi's last known message to Gavin Andresen
    ("Moved on (exact)", "I've moved on to other things"),
    ("Moved on (period)", "I've moved on to other things."),
    ("Good hands Gavin", "It's in good hands with Gavin and everyone."),
    ("Good hands", "It's in good hands"),
    ("Goodbye email", "I wish you wouldn't keep talking about me as a mysterious shadowy figure"),
    ("Shadowy figure", "mysterious shadowy figure"),
    ("Not Dorian", "I am not Dorian Nakamoto"),
    ("Not Dorian (2014)", "I am not Dorian Nakamoto."),

    # ━━━ CATEGORY 16: Anagram Clues ━━━
    ("Anagram: aah it took masons", "aah, it took masons"),
    ("Anagram: aah it took masons v2", "aah it took masons"),
    ("Anagram: Aah, It Took Masons", "Aah, It Took Masons"),
    ("Anagram: A Satoshi Monk", "A Satoshi Monk"),
    ("Anagram: Oath Masons Kin", "Oath Masons Kin"),
    ("Anagram: Took Masonship", "Took Masonship"),
    ("Anagram: NASA Oki Moth", "NASA Oki Moth"),
    ("Anagram: Oaths, OK I'm an", "Oaths, OK I'm an"),
    # Nakamoto meaning in Japanese: "central" or "one who lives in the middle"
    ("Japanese: naka + moto", "中本哲史"),  # Japanese kanji for Satoshi Nakamoto
    ("Japanese: satoshi", "聡"),  # "wise/clever"

    # ━━━ CATEGORY 17: JD2M Reference ━━━
    # Address starting with JD2M, possibly Jack Dorsey + 2 Mint Plaza
    ("JD2M", "JD2M"),
    ("Jack Dorsey 2 Mint", "Jack Dorsey 2 Mint Plaza"),
    ("2 Mint Plaza", "2 Mint Plaza"),
    ("JD2M Bitcoin", "JD2M Bitcoin"),

    # ━━━ CATEGORY 18: Untouched Fortune / 2.56 BTC ━━━
    ("2.56 BTC", "2.56"),
    ("256 satoshis", "256"),
    ("1 million BTC", "1000000"),
    ("Untouched fortune", "untouched"),
    ("1M BTC", "1000000 BTC"),

    # ━━━ CATEGORY 19: Hungarian Notation / Coding Style ━━━
    # Satoshi used Hungarian notation in Bitcoin's C++ code
    ("Hungarian: nNonce", "nNonce"),
    ("Hungarian: nBits", "nBits"),
    ("Hungarian: nTime", "nTime"),
    ("Hungarian: nVersion", "nVersion"),
    ("Hungarian: pszTimestamp", "pszTimestamp"),
    ("Hungarian: pblock", "pblock"),
    ("Hungarian: mapBlockIndex", "mapBlockIndex"),
    ("Hungarian: CBlock", "CBlock"),
    ("Hungarian: CTransaction", "CTransaction"),
    # The actual pszTimestamp variable from the source code
    ("pszTimestamp value", 'const char* pszTimestamp = "The Times 03/Jan/2009 Chancellor on brink of second bailout for banks";'),
    ("pszTimestamp short", "pszTimestamp"),

    # ━━━ CATEGORY 20: Timezone / Location Clues ━━━
    ("GMT+0", "GMT+0"),
    ("London", "London"),
    ("United Kingdom", "United Kingdom"),
    ("UK timezone", "Europe/London"),
    ("British hacker", "British hacker"),

    # ━━━ CATEGORY 21: Deeper Combinations ━━━
    # Name + farewell
    ("Name + moved on", "Satoshi Nakamoto I've moved on to other things"),
    # Genesis + name reversed
    ("Name reversed", "otomakaN ihsotaS"),
    # The specific genesis hex coinbase script prefix
    ("Coinbase script", "04ffff001d0104"),
    # Block 1 timestamp gap (6 days)
    ("6 day gap", "6 days"),
    ("Block 1 gap", "1231469665"),  # Block 1 timestamp
    # Satoshi's P2P Foundation profile said birthday April 5, 1975
    # April 5 = date US Executive Order 6102 (gold confiscation, 1933)
    # 1975 = year private gold ownership re-legalized in US
    ("EO 6102", "Executive Order 6102"),
    ("Gold confiscation", "Executive Order 6102 April 5 1933"),
    ("Gold legal 1975", "Gold legal 1975"),
    ("April 5 1933", "April 5, 1933"),
    ("EO6102 + 1975", "6102 1975"),
    ("Birthday = gold", "April 5 1975 gold"),

    # ━━━ CATEGORY 22: Double-SHA256 style (Satoshi's signature move) ━━━
    # What if the key derivation is SHA256(SHA256(x))?
    # We can't test that with our current function, but try the phrase
    ("SHA256d hint", "SHA-256d"),
    ("Double hash", "double SHA-256"),
    ("Hash of hash", "hash of a hash"),

    # ━━━ CATEGORY 23: Philosophical / Cypherpunk ━━━
    ("Cypherpunk manifesto", "Cypherpunks write code"),
    ("Privacy manifesto", "Privacy is necessary for an open society"),
    ("Eric Hughes quote", "We must defend our own privacy"),
    ("Tim May", "Timothy C. May"),
    ("Wei Dai", "Wei Dai"),
    ("b-money", "b-money"),
    ("Hal Finney", "Hal Finney"),
    ("Hal Finney RIP", "Running bitcoin"),  # Hal's famous first tweet
    ("Hash Cash", "Adam Back Hashcash"),
    ("Nick Szabo", "Nick Szabo"),
    ("Bit Gold", "bit gold"),
    ("Digital gold", "digital gold"),
    ("David Chaum", "David Chaum"),
    ("DigiCash", "DigiCash"),

    # ━━━ CATEGORY 24: The "lad" and casual British references ━━━
    ("Right lad", "right, lad"),
    ("Good lad", "good lad"),
    ("Cheers", "Cheers"),
    ("Brilliant", "Brilliant"),
    ("Bollocks", "bollocks"),
    ("Bloody brilliant", "bloody brilliant"),

    # ━━━ CATEGORY 25: Source code comments & constants ━━━
    ("Coin value", "COIN = 100000000"),
    ("Max money", "MAX_MONEY = 21000000 * COIN"),
    ("Main.cpp line 1", "// Copyright (c) 2009 Satoshi Nakamoto"),
    ("Satoshi copyright", "Copyright (c) 2009 Satoshi Nakamoto"),
    ("MIT license", "Distributed under the MIT/X11 software license"),
    # The famous comment about the longest chain
    ("Longest chain", "The longest chain not only serves as proof of the sequence of events witnessed, but proof that it came from the largest pool of CPU power."),

    # ━━━ CATEGORY 26: Creative multi-clue combos ━━━
    # Combine British + genesis + double space
    ("British genesis dbl", "The Times 03/Jan/2009 Chancellor on brink of second bailout for banks.  Colour me surprised."),
    # Name + anagram
    ("Name + anagram", "Satoshi Nakamoto aah it took masons"),
    # EO6102 + Genesis
    ("EO6102 + Genesis", "Executive Order 6102 The Times 03/Jan/2009"),
    # Birthday meaning
    ("Birthday decoded", "April 5 1975 Executive Order 6102 gold"),
    # Coin + name
    ("1 satoshi", "1 satoshi"),
    ("100M satoshi", "100000000 satoshi"),
    # What if it's literally "One Piece" (treasure reference)
    ("One Piece", "One Piece"),
    ("The One Piece", "The One Piece"),
    ("The One Piece is real", "The One Piece is real"),
    # Pirate-themed (treasure hunt vibe)
    ("Treasure", "X marks the spot"),
    ("Follow the white rabbit", "Follow the white rabbit"),
]

# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  SATOSHI'S TREASURE HUNT")
    print("  Testing", len(CANDIDATES), "candidate passphrases against the blockchain")
    print("=" * 80)
    print()

    hits = []
    errors = []

    for i, (label, candidate) in enumerate(CANDIDATES, 1):
        pk = derive_private_key(candidate)
        pub = private_key_to_public(pk)
        addr = public_key_to_address(pub)

        # Rate limit: Blockstream allows ~10 req/s, be gentle
        time.sleep(0.35)

        tx_count, funded, err = check_address(addr)

        status = ""
        if tx_count is None:
            status = f"  ERROR: {err}"
            errors.append((label, candidate, addr, err))
        elif tx_count > 0:
            balance = (funded - err) / 1e8  # err is 'spent' here
            status = f"  *** HIT *** tx={tx_count}  balance={balance:.8f} BTC"
            hits.append((label, candidate, addr, tx_count, balance))
        else:
            status = "  (empty)"

        short = candidate[:60] + "..." if len(candidate) > 60 else candidate
        print(f"[{i:3d}/{len(CANDIDATES)}] {label}")
        print(f"         String:  \"{short}\"")
        print(f"         Address: {addr}")
        print(f"        {status}")
        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    if hits:
        print(f"\n  {len(hits)} ADDRESS(ES) WITH ACTIVITY FOUND:\n")
        for label, candidate, addr, tx_count, balance in hits:
            print(f"  *** {label} ***")
            print(f"      String:  \"{candidate}\"")
            print(f"      Address: {addr}")
            print(f"      Txs: {tx_count}  Balance: {balance:.8f} BTC")
            print(f"      Explorer: https://blockstream.info/address/{addr}")
            print()
    else:
        print("\n  No addresses with activity found.")

    if errors:
        print(f"\n  {len(errors)} lookup(s) failed (network errors).")

    print(f"\n  Total candidates tested: {len(CANDIDATES)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
