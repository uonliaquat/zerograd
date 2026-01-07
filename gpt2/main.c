#include <stdio.h>
#include <pcre2.h>

int main(){
    printf("Running Main...\n");
    
    PCRE2_SIZE erroffset;
    int errorcode;

    pcre2_code *re = pcre2_compile()


    regex_t regex;
    const char *pattern = "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+";
    const char *subject = "GPT2 was created by OpenAI";
    


    /* Compile regular expression */
    int ret = regcomp(&regex, "'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]+", REG_EXTENDED);
    if(ret == 0){
        printf("RegEx Compiled successfully\n");
    }
    else{
        printf("RegEx Compilation Failed\n");
    }

    /* Executre regular expressing */
    ret = regexec(&regex, "GPT2 was created by OpenAI", 0, NULL, 0);
    if(ret == 0){
        printf("Regex match failed");
    }
    else{
        printf("Match");
    }
    
    regfree(&regex);
    return 0;
}
