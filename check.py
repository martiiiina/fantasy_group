def main():
    import csv

    file1 = "Train_def_0"
    file2 = "Train_def"

    with open(file1, newline='') as f1, open(file2, newline='') as f2:
        reader1 = list(csv.reader(f1))
        reader2 = list(csv.reader(f2))

    if reader1 == reader2:
        print("✅ I due file CSV sono IDENTICI.")
    else:
        print("❌ I file CSV sono diversi.")

        # Mostra differenze nelle prime righe non uguali
        for i, (row1, row2) in enumerate(zip(reader1, reader2)):
            if row1 != row2:
                print(f"🔹 Differenza alla riga {i+1}:")
                print(f"  {file1}: {row1}")
                print(f"  {file2}: {row2}")
                break

if __name__ == "__main__":
    main()
