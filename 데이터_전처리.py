{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "데이터 전처리.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOsZC6HiV9k/7uL66j4LQKj",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Jibin-Song/self-study_machine-learning/blob/main/%EB%8D%B0%EC%9D%B4%ED%84%B0_%EC%A0%84%EC%B2%98%EB%A6%AC.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "04SAt5hUXqPF"
      },
      "source": [
        "# 작성 날짜: 2021.08.19(목)\n",
        "# 프로그램 개요: 데이터 전처리; 표준편차를 이용하여 데이터 스케일 맞추기"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_W_sH-nOHOLT"
      },
      "source": [
        "#데이터 전처리\n",
        "\n",
        "fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, \n",
        "                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, \n",
        "                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, \n",
        "                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]\n",
        "fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, \n",
        "                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, \n",
        "                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, \n",
        "                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]\n",
        "\n",
        "#입력 데이터와 타깃 데이터 만들기\n",
        "import numpy as np\n",
        "\n",
        "fish_input = np.column_stack((fish_length, fish_weight))\n",
        "fish_target = np.concatenate((np.ones(35), np.zeros(14)))\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "t9IrUTaOJn9g"
      },
      "source": [
        "#사이킷런으로 훈련세트와 테스트 세트 나누기\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, stratify=fish_target)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3tF4oBL_KaxS",
        "outputId": "b9de9190-9ff4-4fc6-8d4d-3a0a5a27216a"
      },
      "source": [
        "#훈련하기\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "kn = KNeighborsClassifier()\n",
        "kn.fit(train_input, train_target)\n",
        "kn.score(test_input, test_target)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {},
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MlVNh5IjLoug",
        "outputId": "87d3d184-f127-47a7-92b7-da2540a3f4c2"
      },
      "source": [
        "print(kn.predict([[25, 150]]))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[0.]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1OUCKweRQGBU"
      },
      "source": [
        "#이웃하는 n개의 거리와, 인덱스 반환\n",
        "distances, indexes = kn.kneighbors([[25,150]])\n",
        "print(distances, indexes)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "R45_DPcAL5Z0",
        "outputId": "d7332a78-181e-4cac-a5ca-5d6d17e045ce"
      },
      "source": [
        "#산점도 그리기\n",
        "import matplotlib.pyplot as plt\n",
        "plt.scatter(train_input[:,0], train_input[:,1])\n",
        "plt.scatter(25,150,marker='^')\n",
        "plt.scatter(train_input[indexes,0], train_input[indexes,1],marker='D')\n",
        "plt.xlabel('length')\n",
        "plt.ylabel('weight')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "WLeTgR_GMTVf",
        "outputId": "cbf80d75-6c04-4ad5-c862-c8f7627c9248"
      },
      "source": [
        "#기준을 다시 맞춰라\n",
        "import matplotlib.pyplot as plt\n",
        "plt.scatter(train_input[:,0], train_input[:,1])\n",
        "plt.scatter(25,150,marker='^')\n",
        "plt.scatter(train_input[indexes,0], train_input[indexes,1],marker='D')\n",
        "plt.xlabel('length')\n",
        "plt.ylabel('weight')\n",
        "#축의 범위를 수동으로 지정\n",
        "plt.xlim((0,1000))\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HvwEUTWAQygC",
        "outputId": "e45f75b5-9b08-4666-ddb8-40fe8ca5cc0d"
      },
      "source": [
        "#표준점수 이용하여 스케일 맞추기\n",
        "mean = np.mean(train_input, axis=0)\n",
        "std = np.std(train_input, axis=0)\n",
        "print(mean, std)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[ 26.79722222 437.9       ] [ 10.17200641 330.74621607]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6b1qEMj4RO2o"
      },
      "source": [
        "#브로드캐스팅\n",
        "train_scaled = (train_input - mean) / std"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pYXPZWohR6Fp",
        "outputId": "035f23be-f7a3-499f-ea38-fd879a65db7f"
      },
      "source": [
        "#예측할 샘플도 스케일 맞춰주기\n",
        "new = ([25, 150] - mean) / std\n",
        "print(new)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[-0.17668316 -0.87045591]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "blhpuS69RlgM"
      },
      "source": [
        "#다시 훈련하기\n",
        "kn.fit(train_scaled,train_target)\n",
        "test_scaled = (test_input - mean) / std\n",
        "kn.score(test_scaled, test_target)\n",
        "\n",
        "#이웃 인덱스 다시 구하기\n",
        "distances, indexes = kn.kneighbors([new])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "hbASRPU7R2Eg",
        "outputId": "64e059b9-beca-47be-cb22-1322111042ff"
      },
      "source": [
        "#산점도 다시 구하기\n",
        "plt.scatter(train_scaled[:,0], train_scaled[:,1])\n",
        "plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1])\n",
        "plt.scatter(new[0], new[1], marker='^')\n",
        "plt.xlabel('length')\n",
        "plt.ylabel('weight')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAY0AAAEGCAYAAACZ0MnKAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAY6UlEQVR4nO3df3Bc5X3v8c8HRSTbJEUQOzEWTgyJx5TiFoMgP+hkSJvGQFvbdZIBtzOFNh3T9jLN7Z1qxr7xDYShNa2n7TQ33BqHkJJJS2ipI0zjVgm/Jp3emGs5thEO1cWhSewFYpFUJFx0Qcjf/rFHZiXvjyNpd8/R7vs1s6NznnN89nskS999nuc8z+OIEAAAaZyWdQAAgIWDpAEASI2kAQBIjaQBAEiNpAEASO11WQfQaIsWLYrly5dnHQYALCj79+9/PiIW1zuv7ZLG8uXLNTQ0lHUYALCg2P5umvNongIApEbSAACkRtIAAKRG0gAApEbSAACk1nZPTwFAow0cKGr74IieGRvX0p6C+tes1PrVvVmHlQmSBgDUMHCgqC27hjU+MSlJKo6Na8uuYUnqyMRB8xQA1LB9cORkwpgyPjGp7YMjGUWULZIGANTwzNj4rMrbHUkDAGpY2lOYVXm7I2kAQA39a1aq0N01razQ3aX+NSsziuhUAweKuvy2h3Xu5q/o8tse1sCBYtPei45wAKhhqrM7r09PtbqjnqQBAHWsX92bmyQxU62O+mbETPMUACxgre6op6YBAE3U7IGBS3sKKlZIEM3qqKemAQBNMtXfUBwbV+i1/oZGdlT3r1mp7tM8raz7NDeto56kAQBN0rKBga6z30AkDQBoklb0N2wfHNHEZEwrm5iMpo1YzzRp2L7L9nHbT1Q5foXtF2wfTF6fbHWMADBXrRgY2OqO8KxrGn8t6co65/xLRFyUvG5pQUwA0BCtGBjY6hHrmSaNiPi6pB9mGQMANMv61b3atmGVensKsqTenoK2bVjV0KenWj1ifSE8cvte24ckPSPpDyPi8MwTbG+StEmS3v72t7c4PACortkDA1s9Yt0RUf+sJrK9XNI/RsSFFY79pKQTEfGi7asl/WVErKh1vb6+vhgaGmpKrADQrmzvj4i+eudl3adRU0T8KCJeTLb3SOq2vSjjsACgY+U6adheYtvJ9mUqxfuDbKMCgM6VaZ+G7XskXSFpke1jkm6S1C1JEbFD0kck/a7tVyWNS7o2sm5PA4AOlmnSiIiNdY5/RtJnWhQOAKCOXDdPAQDyhaQBAEiNpAEASI2kAQBIjaQBAEiNpAEASI2kAQBIjaQBAEhtIcxyCwBNMXCg2LLZYdsFSQNARxo4UNSWXcMn1/Aujo1ry65hSSJx1EDzFICOtH1w5GTCmDI+Mdm0tbXbBUkDQEdq9dra7YKkAaAjtXpt7XZB0gDQkVq9tna7oCMcQEdq9dra7YKkAaBjrV/dS5KYJZqnAACpkTQAAKmRNAAAqZE0AACpkTQAAKmRNAAAqZE0AACpkTQAAKllmjRs32X7uO0nqhy37U/bPmL7cdsXtzpGAMBrsq5p/LWkK2scv0rSiuS1SdJftSAmAEAVmSaNiPi6pB/WOGWdpC9EyV5JPbbPbk10AICZsq5p1NMr6WjZ/rGkbBrbm2wP2R4aHR1tWXAA0GnynjRSiYidEdEXEX2LFy/OOhwAaFt5TxpFScvK9s9JygAAGch70tgt6TeSp6jeI+mFiHg266AAoFNlup6G7XskXSFpke1jkm6S1C1JEbFD0h5JV0s6IuklSb+ZTaQAACnjpBERG+scD0n/pUXhAADqYOU+AG1n4ECRZVybhKQBoK0MHCiq/75DmpgMSVJxbFz99x2SJBJHA+S9IxwAZuVTDxw+mTCmTEyGPvXA4Ywiai8kDQBt5T9emphVOWaHpAEASI0+DQCZ2zowrHseO6rJCHXZ2vjuZbp1/ao5Xaun0K2x8VNrFT2F7vmGCVHTAJCxrQPD+uLe72kySv0QkxH64t7vaevA8Jyu98s/W3lO02rlmB2SBoBM3fPY0VmV1/PIv1WetLRaOWaH5ikAmZqqYaQpTzP+4pmx8YrXq1aO2aGmASBTXXaq8oEDRW3ZNazi2LhCpfEXW3YNa+DA9DlMl/YUKl6vWjlmh6QBIFMb370sVfn2wRGNT0xOKxufmNT2wZFpZf1rVqrQ3TWtrNDdpf41KxsQLWieApCpqaek6j09lbbZaaq5imlEmoOkASBzt65fVfcR26U9BRUrJI5KzU7rV/eSJJqE5ikACwLNTvlATQPAgkCzUz6QNADkUrXHa0kS2SJpAMidqcdrp56Wmnq8VmJ686zRpwEgd9I+XovWI2kAyB1GdecXSQNA7jCqO79IGgByh8dr84uOcAC5w+O1+UXSAJBLPF6bTyQNALmUZhp0tF6mfRq2r7Q9YvuI7c0Vjl9ve9T2weT121nECaC10k6DjtbLrKZhu0vS7ZJ+UdIxSfts746Ib8049d6IuLHlAQKYt7mu/V1rnAa1jWxlWdO4TNKRiHg6Il6R9CVJ6zKMB0ADzWftb8Zp5FeWSaNXUvkiwMeSspk+bPtx2/fZrrhai+1NtodsD42Osg4wkAfzWfubcRr5lfdxGg9IWh4RPyPpa5LurnRSROyMiL6I6Fu8eHFLAwRQ2WzW/p6JcRr5lWXSKEoqrzmck5SdFBE/iIiXk907JV3SotgAzFPatb8rWb+6V9s2rFJvT0GW1NtT0LYNq+jPyIEsH7ndJ2mF7XNVShbXSvq18hNsnx0Rzya7ayU92doQAczVxncv0xf3fq9ieRqVxmnwGG72MksaEfGq7RslDUrqknRXRBy2fYukoYjYLen3ba+V9KqkH0q6Pqt4AcxO2rW/02K69HxwpGhfXEj6+vpiaGgo6zAANNjltz1ccY3w3p6C/nXzz2cQUXuxvT8i+uqdl6pPw/bH05QBQLPwGG4+pO0Iv65C2fUNjAMAauIx3HyomTRsb7T9gKRzbe8uez2iUh8DALQEj+HmQ72O8P8t6VlJiyT9WVn5jyU93qygAGAmpkvPBzrCAQAN7wjfYPsp2y/Y/pHtH9v+0fzDBAAsJGnHafyppF+JCAbXAUAHS/v01PdJGACAmjUN2xuSzSHb90oakDQ1F5QiYlcTYwMA5Ey95qlfKdt+SdKHyvZDEkkDADpIzaQREb/ZqkAAAPmXqiPc9qcrFL+g0sSC9zc2JABAXqV9euoNks6X9PfJ/ocl/bukn7X9gYj4r80IDkA+MUV550qbNH5G0uURMSlJtv9K0r9I+jlJ9Rf8BdA2mKK8s6V95PZMSW8q23+jpLOSJPJy5X8CoB1tHxw5mTCmjE9MavvgSEYRoZVmM7jvoO1HJVnS+yX9se03SnqwSbEByCGmKO9sqZJGRHzO9h5JlyVF/z0inkm2+5sSGYBcWtpTqLgYElOUd4Z6U6Ofn3y9WNLZko4mryVJGYAO079mpbpP87Sy7tPMFOUdol5N479J2qTp06JPCUmssQh0ItfZR9uqN7hvU/L1A60JB0DebR8c0cTk9CUVJiZD2wdHeHqqA6SdGv0nbG+1vTPZX2H7l5sbGoA8oiO8s6V95Pbzkl6R9L5kvyjp1qZEBCDXWKu7s6V95PadEXGN7Y2SFBEv2aYVE2gD5aO7zyh0y5bGXpqoOtK7f83KaYP7JNbq7iRpk8YrtgsqdX7L9jvFoD5gwZs5untsfOLksWojvVmru7OlTRo3SfpnScts/42kyyVdP983t32lpL+U1CXpzoi4bcbx10v6gqRLJP1A0jUR8Z35vi+Akkqju8tNjfSemRDWr+4lSXSotH0a10n6iqRbJP2tpL6IeHQ+b2y7S9Ltkq6SdIGkjbYvmHHaxyT9R0S8S9JfSPqT+bwngOnSdF7TwY1yaZPG51Sa6XatpP8p6Q7bH5/ne18m6UhEPB0Rr0j6kqR1M85ZJ+nuZPs+Sb9AXwowe/t236Hnbn6XTtx0hp67+V3at/sOSek6r+ngRrlUSSMiHpH0R5L+h6TPSuqT9LvzfO9elUaXTzmWlFU8JyJeVWkNj7fMvJDtTbaHbA+Njo7OMyygvezbfYcu3L9VSzSq0ywt0agu3L9V+3bfof41K1Xo7qr6b+ngxkxpx2k8JOlfJV0jaUTSpRFxfjMDm42I2BkRfRHRt3jx4qzDAXJl2Te3q+BXppUV/IqWfXO71q/u1bYNq9TbU5Al9RS6deZPdMuSensK2rZhFX0XmCZtR/jjKnVGX6jSp/0x29+IiPk0dhYlLSvbPycpq3TOMduvk3SGSh3iAFJ6a4xWnObjrfG8JDq1MTtpm6f+ICLeL2mDSn+0Py9pbJ7vvU/SCtvn2j5d0rWSds84Z7dKnfCS9BFJD0dECEBqx1259n3ci1ocCdpB2uapG23fK+mASp3Td6n01NOcJX0UN0oalPSkpL+LiMO2b7G9Njntc5LeYvuISpMnbp7PewKd6OjF/RqP06eVjcfpOnoxqxpg9mazRvifS9qf/LFviIjYI2nPjLJPlm3/f0kfbdT7AZ3o0rU3aJ9KfRtvjed13It09JJ+Xbr2hqxDwwLkdmvt6evri6GhoazDAIAFxfb+iOird17amgaADrBv9x1JjWRUx71YRy+mRoLpSBoAJL02nqPgV6RkPMcZ+7dqn0TiwElpR4QDaHO1xnMAU0gaACQl4zkqlj/f4kiQZyQNAJIYz4F0SBpAB6g2YWE5xnMgDTrCgTaXtoOb8RxIg3EaQJt77uZ3aYlO7a94Tou15OYjGUSEPEo7ToPmKaDN0cGNRiJpAG2ODm40EkkDaHNZdnCn6YDHwkJHONDmsurgZoR5e6IjHEBT0AG/sNARDiBTdMC3J5IGgKagA749kTQANAUjzNsTSQNAU1y69gY9ccmtek6LdSKs57RYT1xyK53gCxwd4QAAOsIBAI1H0gAApEbSAACkRtIAAKRG0gAApJZJ0rB9lu2v2X4q+XpmlfMmbR9MXrtbHScAYLqsahqbJT0UESskPZTsVzIeERclr7WtCw8AUElWSWOdpLuT7bslrc8oDgDALGSVNN4WEc8m289JeluV895ge8j2XttVE4vtTcl5Q6OjlSdJAwDMX9PW07D9oKQlFQ59onwnIsJ2tWHp74iIou3zJD1sezgivj3zpIjYKWmnVBoRPs/QAQBVNC1pRMQHqx2z/X3bZ0fEs7bPlnS8yjWKydenbT8qabWkU5IGAKA1smqe2i3pumT7Okn3zzzB9pm2X59sL5J0uaRvtSxCYB5Y5hTtKqukcZukX7T9lKQPJvuy3Wf7zuScn5I0ZPuQpEck3RYRJA3k3tQyp0s0qtOSZU4v3L+VxIG2wCy3QIOxzCkWIma5BTLCMqdoZyQNoMFY5hTtjKQBNBjLnKKdkTSABmOZU7QzOsIBAHSEAwAaj6QBAEiNpAEASI2kAQBIjaQBAEiNpAEASI2kAQBIjaQBAEiNpAEASI2kAQBIjaQBAEiNpAE0yOhLo7rqH67S8+Osm4H2RdIAGmTH4ztUfLGoHYd2ZB0K0DQkDaABRl8a1f1H7lcoNHBkgNoG2hZJA2iAHY/v0Ik4IUk6ESeobaBtkTSAeZqqZUycmJAkTZyYoLaBtkXSAOapvJYxhdoG2hVJA5inR48+erKWMWXixIQeOfpIRhEBzfO6rAMAFrqHPvpQ1iEALZNJTcP2R20ftn3CdtU1aW1faXvE9hHbm1sZIwDgVFk1Tz0haYOkr1c7wXaXpNslXSXpAkkbbV/QmvAAAJVk0jwVEU9Kku1ap10m6UhEPJ2c+yVJ6yR9q+kBAgAqynNHeK+ko2X7x5KyU9jeZHvI9tDo6GhLggOATtS0mobtByUtqXDoExFxfyPfKyJ2StopSX19fdHIawMAXtO0pBERH5znJYqSlpXtn5OUAQAykufmqX2SVtg+1/bpkq6VtDvjmACgo2X1yO2v2j4m6b2SvmJ7MClfanuPJEXEq5JulDQo6UlJfxcRh7OIFwBQktXTU1+W9OUK5c9Iurpsf4+kPS0MDQBQQ56bpwAAOUPSAACkRtIAAKRG0gAApEbSAACkRtIAAKRG0gAApEbSAACkxsp9OTdwoKjtgyN6ZmxcS3sK6l+zUutX99Y9BgDNQNLIsYEDRW3ZNazxiUlJUnFsXFt2DZ88Xu0YiQNAs5A0apj6JF8cG1eXrckI9ZZ9ot86MKx7HjuqyXhtNvZK5828XtqawfbBkZNJYcr4xKS2D46c3K50jKQBoFlIGomZf9A/cP5i/cP+4sk/zFOJoTg2rj+496Buf+QpPXX8/51ynfLzyj/516o1VPsj/8zY+KzK6x0DgPmiI1yvNQMVx8YVKv1B/5u93zvlk/yUkComjJnKawX1ag2VLO0pVC2vdQwAmoWkocp/0Bu1/N/UJ/+51Br616xUobtrWlmhu0v9a1bWPAYAzULzlJrbpDP1yX9pT0HFCu9Tq2Yw1WxVqx+Ep6cAtBJJQ9X/oFvzq3GUf/LvX7NyWp/GzOPVrF/dWzUR1DoGAM1A85SqNwP9+nvert4qNYFCd5cuf+dZ6rKnlU/t9/YUtG3DqpN/1Nev7tW2DavU21OQKxwHgIWAmobSNQM1YiAdNQMAC50jGtXlmw99fX0xNDSUdRgAsKDY3h8RffXOo3kKAJAaSQMAkBpJAwCQGkkDAJAaSQMAkFrbPT1le1TSd1vwVoskPd+C92kF7iV/2uU+JO4lr2beyzsiYnG9f9R2SaNVbA+leTxtIeBe8qdd7kPiXvJqrvdC8xQAIDWSBgAgNZLG3O3MOoAG4l7yp13uQ+Je8mpO90KfBgAgNWoaAIDUSBoAgNRIGinZ/qjtw7ZP2K76mJrt79getn3Qdi6n253FvVxpe8T2EdubWxljWrbPsv01208lX8+sct5k8jM5aHt3q+Ospt732Pbrbd+bHH/M9vLWR5lOinu53vZo2c/ht7OIsx7bd9k+bvuJKsdt+9PJfT5u++JWx5hWinu5wvYLZT+TT9a9aETwSvGS9FOSVkp6VFJfjfO+I2lR1vHO914kdUn6tqTzJJ0u6ZCkC7KOvUKcfyppc7K9WdKfVDnvxaxjncv3WNLvSdqRbF8r6d6s457HvVwv6TNZx5riXt4v6WJJT1Q5frWkf1Jpcc/3SHos65jncS9XSPrH2VyTmkZKEfFkRIxkHUcjpLyXyyQdiYinI+IVSV+StK750c3aOkl3J9t3S1qfYSyzleZ7XH5/90n6BXvGcpH5sFD+v9QVEV+X9MMap6yT9IUo2Supx/bZrYludlLcy6yRNBovJH3V9n7bm7IOZh56JR0t2z+WlOXN2yLi2WT7OUlvq3LeG2wP2d5rOy+JJc33+OQ5EfGqpBckvaUl0c1O2v8vH06adO6zvaw1oTXcQvndSOu9tg/Z/ifbP13vZJZ7LWP7QUlLKhz6RETcn/IyPxcRRdtvlfQ12/+WZPuWatC95EKteynfiYiwXe0Z8nckP5fzJD1sezgivt3oWFHTA5LuiYiXbd+gUg3q5zOOqdN9U6XfjRdtXy1pQNKKWv+ApFEmIj7YgGsUk6/HbX9ZpWp7y5NGA+6lKKn8k+A5SVnL1boX29+3fXZEPJs0ERyvco2pn8vTth+VtFqlNvgspfkeT51zzPbrJJ0h6QetCW9W6t5LRJTHfadK/VELUW5+N+YrIn5Utr3H9v+yvSgiqk7KSPNUA9l+o+03T21L+pCkik8tLAD7JK2wfa7t01XqhM3NU0dldku6Ltm+TtIptSjbZ9p+fbK9SNLlkr7VsgirS/M9Lr+/j0h6OJIezJypey8z2v3XSnqyhfE10m5Jv5E8RfUeSS+UNZEuKLaXTPWR2b5MpZxQ+0NJ1r37C+Ul6VdVart8WdL3JQ0m5Usl7Um2z1PpqZFDkg6r1BSUeexzuZdk/2pJ/1elT+R5vZe3SHpI0lOSHpR0VlLeJ+nOZPt9koaTn8uwpI9lHXet77GkWyStTbbfIOnvJR2R9H8knZd1zPO4l23J78UhSY9IOj/rmKvcxz2SnpU0kfyefEzS70j6neS4Jd2e3OewajxNmfUrxb3cWPYz2SvpffWuyTQiAIDUaJ4CAKRG0gAApEbSAACkRtIAAKRG0gAApEbSAGbJ9otNuOZFyYjcqf2bbf9ho98HmC+SBpAPF6k0zgHINZIGMA+2+23vSybh+1RSttz2k7Y/m6xb8lXbheTYpcm5B21vt/1EMoL6FknXJOXXJJe/wPajtp+2/fsZ3SIwDUkDmCPbH1JpcrfLVKopXGL7/cnhFZJuj4ifljQm6cNJ+ecl3RARF0malKQoTSX+SZXWyrgoIu5Nzj1f0prk+jfZ7m7BbQE1kTSAuftQ8jqg0myh5+u1GUL/PSIOJtv7JS233SPpzRHxjaT8b+tc/ysR8XKUJo87rurTvgMtwyy3wNxZ0raIuGNaYWlJ1pfLiiYlFeZw/ZnX4PcVmaOmAczdoKTfsv0mSbLdm6yjUlFEjEn6se13J0XXlh3+saQ3Ny1SoEFIGsAcRcRXVWpi+obtYZWWY633h/9jkj5r+6CkN6q0Ep9UmvX1ghkd4UDuMMst0EK23xQRLybbmyWdHREfzzgsIDXaSIHW+iXbW1T63fuupOuzDQeYHWoaAIDU6NMAAKRG0gAApEbSAACkRtIAAKRG0gAApPafNQt9d3oAViYAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}