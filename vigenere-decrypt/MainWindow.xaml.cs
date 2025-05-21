using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace vigenere_decrypt;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private void Decrypt_Click(object sender, RoutedEventArgs e) {
        string key = KeyBox.Text;
        string cipherText = CipherTextBox.Text;
        string result = VigenereUtils.Decrypt(cipherText, key);
        PlainTextBox.Text = result;
    }

    private void Encrypt_Click(object sender, RoutedEventArgs e) {
        string key = KeyBox.Text;
        string plainText = CipherTextBox.Text;
        string result = VigenereUtils.Encrypt(plainText, key);
        PlainTextBox.Text = result;
    }

}

public static class VigenereUtils {
    public static string Encrypt(string ciphertext, string key) {
        ciphertext = ciphertext.ToUpper();
        key = key.ToUpper();

        StringBuilder result = new StringBuilder();
        int keyIndex = 0;
        foreach (char c in ciphertext) {
            if (char.IsLetter(c)) {
                char keyChar = key[keyIndex % key.Length];
                char encryptedChar = (char)((((c - 'A') + (keyChar - 'A')) % 26) + 'A');
                result.Append(encryptedChar);
                keyIndex++;
            } else {
                result.Append(c);
            }
        }
        return result.ToString();
    }

    public static string Decrypt(string ciphertext, string key) {
        ciphertext = ciphertext.ToUpper();
        key = key.ToUpper();

        StringBuilder result = new StringBuilder();
        int keyIndex = 0;
        foreach (char c in ciphertext) {
            if (char.IsLetter(c)) {
                char keyChar = key[keyIndex % key.Length];
                char decryptedChar = (char)((((c - 'A') - (keyChar - 'A') + 26) % 26) + 'A');
                result.Append(decryptedChar);
                keyIndex++;
            } else {
                result.Append(c);
            }
        }
        return result.ToString();
    }
}

public static class BreakVigenere {
    public static string Break(string ciphertext) {
        // This is a placeholder for the actual Vigenere breaking algorithm.
        return "Breaking Vigenere is not implemented.";
    }

    public static int[] GetCoincidenceFrequency(string ciphertext) {
        int[] coincidences = new int[100];
        for (int i = 0; i < ciphertext.length; i++) {
            for (int j = i + 1; j < ciphertext.length; j++) {
                if (ciphertext[i] == ciphertext[j]) {
                    coincidences[j - i]++;
                }
            }
        }
        return coincidences;
    }

    public static int GetKeyLength(int[] coincidences) {
        // Placeholder
    }
}