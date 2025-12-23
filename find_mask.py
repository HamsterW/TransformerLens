tokenizer = tl_model.tokenizer

print(f"--- Tokenizer Check ---")
print(f"1. Reported Mask Token ID: {tokenizer.mask_token_id}")
print(f"2. Reported Mask Token:    '{tokenizer.mask_token}'")

# Check the specific ID 126336
target_id = 126336
decoded = tokenizer.decode([target_id])
print(f"\n--- Checking ID {target_id} ---")
print(f"ID {target_id} decodes to: '{decoded}'")

# Safety Check
if tokenizer.mask_token_id is None:
    print("\n⚠️ WARNING: Tokenizer does not have a mask_token_id set!")
    print(f"You should manually set: MASK_ID = {target_id}")
elif tokenizer.mask_token_id != target_id:
    print(f"\n⚠️ MISMATCH: Config says {target_id} but tokenizer says {tokenizer.mask_token_id}")
    print("Trust the '126336' if you are using the official GSAI-ML weights.")
else:
    print("\n✅ SUCCESS: Tokenizer and Config agree on the mask ID.")